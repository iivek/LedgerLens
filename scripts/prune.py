"""Weekend 3: structured depth pruning (ShortGPT-style layer drop) + recovery SFT.

Importance = 1 - cos(layer input, layer output) on calibration data: layers that
barely rotate the residual stream are cheapest to remove. Drops the k least
important LLM decoder layers (vision tower untouched), then a short recovery
fine-tune (rerun train_sft.py on the pruned checkpoint) restores most accuracy.

    uv run --extra gpu python scripts/prune.py \
        --model runs/sft-2b-teacher-agree-merged --drop 6 \
        --calib data/distill/cord_train_teacher.jsonl --out models/student2b-prune6
    # then: python scripts/train_sft.py --student models/student2b-prune6 ... (recovery)
"""

from __future__ import annotations

import argparse
import json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--drop", type=int, default=6, help="number of decoder layers to remove")
    ap.add_argument("--calib", required=True, help="teacher_label.py jsonl for importance estimation")
    ap.add_argument("--n-calib", type=int, default=64)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    llm = model.model.language_model  # Qwen2-VL: text decoder stack
    layers = llm.layers
    n_layers = len(layers)

    texts = []
    with open(args.calib) as f:
        for line in f:
            rec = json.loads(line)
            texts.append(rec["prompt"] + "\n" + json.dumps(rec["gold"], ensure_ascii=False))
    texts = texts[: args.n_calib]

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
        for t in texts:
            ids = tokenizer(t, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
            model.model.language_model(input_ids=ids["input_ids"])
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
    model.config.text_config.num_hidden_layers = len(kept) \
        if hasattr(model.config, "text_config") else len(kept)
    if hasattr(model.config, "num_hidden_layers"):
        model.config.num_hidden_layers = len(kept)

    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    AutoProcessor.from_pretrained(args.model).save_pretrained(args.out)
    with open(f"{args.out}/prune_report.json", "w") as f:
        json.dump({"dropped_layers": drop_idx, "importance": importance.tolist()}, f, indent=2)
    print(f"pruned model ({n_layers}->{len(kept)} layers) -> {args.out}; run recovery SFT next")


if __name__ == "__main__":
    main()
