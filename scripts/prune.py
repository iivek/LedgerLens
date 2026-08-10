"""Weekend 3: structured depth pruning (ShortGPT-style layer drop) + recovery SFT.

Importance = 1 - cos(layer input, layer output) on calibration data: layers that
barely rotate the residual stream are cheapest to remove. Drops the k least
important LLM decoder layers (vision tower untouched), then a short recovery
fine-tune (rerun train_sft.py on the pruned checkpoint) restores most accuracy.

Calibration runs FULL multimodal forwards (image + text, via the real model
call, not just the language_model submodule on text-only input_ids) so image
embeddings actually flow through the layers being measured. An earlier
text-only version of this script would rank layers by behavior on a
distribution the model never sees at deployment -- for this task, every real
input includes ~500-1500 image tokens dominating the sequence, so a layer
whose job is consuming visual embeddings could look unimportant on text-only
calibration and get dropped. Caught by independent review, fixed here.

    uv run --extra gpu python scripts/prune.py \
        --model runs/sft-2b-teacher-agree-merged --drop 6 \
        --calib data/distill/cord_train_teacher_0_200.jsonl \
        --image-root /kaggle/input/datasets/<account>/ledgerlens-train-data \
        --out models/student2b-prune6
    # then: python scripts/train_sft.py --student models/student2b-prune6 ... (recovery)
"""

from __future__ import annotations

import argparse
import json
import random


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--drop", type=int, default=6, help="number of decoder layers to remove")
    ap.add_argument("--calib", required=True, help="teacher_label.py jsonl for importance estimation")
    ap.add_argument("--image-root", required=True,
                    help="directory each --calib record's relative image path resolves against "
                         "-- teacher_label.py stores image paths relative to the mounted "
                         "train-data root, not absolute, same convention as train_sft.py")
    ap.add_argument("--n-calib", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).parent.parent / "src"))

    import torch
    import torch.nn.functional as F
    from qwen_vl_utils import process_vision_info
    from transformers import AutoModelForImageTextToText, AutoProcessor

    from ledgerlens.schema import MAX_PIXELS, MIN_PIXELS

    # T4 (Turing, sm_75) has no bf16 support -- same fix as train_sft.py/
    # qat_torchao.py, missed here in an earlier pass. Also missing here
    # originally: min/max_pixels, which left this on whichever default the
    # library ships, diverging from every other stage's pixel budget.
    #
    # device_map="auto" is wrong here, confirmed by a real crash: Kaggle's
    # "NvidiaTeslaT4" machine shape provides two physical T4s, and "auto"
    # sharded this ~2.2B model's decoder layers across both -- the importance
    # hook's `hin`/`hout` (input/output of one layer) ended up on different
    # cuda: devices, crashing cosine_similarity. The model fits fp16 on a
    # single T4 with room to spare (its own merged checkpoint is 4.4GB), so
    # there's no reason to shard it; pin to one GPU instead of teaching the
    # hook to move tensors across a split it doesn't need.
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map={"": 0}
    )
    processor = AutoProcessor.from_pretrained(args.model, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
    llm = model.model.language_model  # Qwen2-VL: text decoder stack
    layers = llm.layers
    n_layers = len(layers)

    calib_recs = []
    with open(args.calib) as f:
        for line in f:
            calib_recs.append(json.loads(line))
    random.Random(args.seed).shuffle(calib_recs)  # don't silently calibrate on one file's ordering
    calib_recs = calib_recs[: args.n_calib]

    # Hook every decoder layer; accumulate mean (1 - cosine) between its in/out states
    importance = torch.zeros(n_layers)
    counts = torch.zeros(n_layers)
    hooks = []

    def make_hook(idx):
        def hook(module, inputs, output):
            hin = inputs[0]
            hout = output[0] if isinstance(output, tuple) else output
            cos = F.cosine_similarity(hin.float(), hout.float(), dim=-1).mean()
            importance[idx] += (1 - cos).item()
            counts[idx] += 1
        return hook

    for i, layer in enumerate(layers):
        hooks.append(layer.register_forward_hook(make_hook(i)))
    with torch.inference_mode():
        for rec in calib_recs:
            messages = [{"role": "user", "content": [
                {"type": "image", "image": f"file://{_Path(args.image_root) / rec['image']}"},
                {"type": "text", "text": rec["prompt"]},
            ]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, _ = process_vision_info(messages)
            inputs = processor(text=[text], images=image_inputs, return_tensors="pt").to(model.device)
            model(**inputs)  # full multimodal forward -- image tokens reach every hooked layer
    for h in hooks:
        h.remove()

    importance /= counts.clamp(min=1)
    drop_idx = sorted(importance.argsort()[: args.drop].tolist())
    print(f"layer importance (1-cos): {[f'{v:.4f}' for v in importance.tolist()]}")
    print(f"dropping layers {drop_idx} of {n_layers}")

    kept = torch.nn.ModuleList([l for i, l in enumerate(layers) if i not in drop_idx])
    for new_idx, layer in enumerate(kept):  # keep rotary/cache indexing consistent
        layer.self_attn.layer_idx = new_idx
    llm.layers = kept
    # Qwen2-VL nests the text config; update whichever layout this transformers
    # version uses, or both. Mismatched config vs. state_dict layer counts don't
    # hard-fail on reload — from_pretrained warns and reconstructs extra/random
    # layers, silently producing a broken checkpoint, so both must be kept in sync.
    if hasattr(model.config, "text_config"):
        model.config.text_config.num_hidden_layers = len(kept)
    if hasattr(model.config, "num_hidden_layers"):
        model.config.num_hidden_layers = len(kept)

    model.save_pretrained(args.out)
    # Save the processor instance already in use (min/max_pixels set above),
    # not a freshly re-fetched default one -- otherwise the saved checkpoint's
    # preprocessor_config.json silently reverts to library defaults, diverging
    # from the pixel budget this model was actually calibrated/will be
    # recovery-tuned/evaluated under. Also fixes a real crash: `tokenizer` was
    # referenced here but never defined anywhere in this script (would have
    # failed with NameError right after the expensive calibration forward
    # passes and layer-drop already completed).
    processor.save_pretrained(args.out)
    with open(f"{args.out}/prune_report.json", "w") as f:
        json.dump({"dropped_layers": drop_idx, "importance": importance.tolist()}, f, indent=2)
    print(f"pruned model ({n_layers}->{len(kept)} layers) -> {args.out}; run recovery SFT next")


if __name__ == "__main__":
    main()
