"""Weekend 3: QAT via torchao on the LoRA-merged student (int8 dyn-act / int4 weight).

Strategy per spec: bound compute by running QAT as a *short* fine-tune with fake
quantization on the LLM linear layers, reusing the SFT data pipeline.

    uv run --extra gpu python scripts/qat_torchao.py \
        --model runs/sft-2b-teacher-agree-merged \
        --data data/distill/cord_train_teacher_*.jsonl --image-root <train-data-mount> \
        --policy gold --steps 500 --out models/student2b-qat-int4

KNOWN UNRESOLVED ISSUES (flagged by review, not yet fixed -- needs a scope
decision before this is trustworthy, not just a bug fix):

1. Int8DynActInt4WeightQATQuantizer simulates int8-dynamic-activation +
   int4-weight quantization. The actual deployment quantizer is llama.cpp's
   Q4_K_M -- a weight-only, blockwise k-quant scheme. These are NOT the same
   quantization scheme, so "QAT beats PTQ" is not a claim this script can
   support as-is: the fake-quant during training doesn't match what the model
   actually gets quantized to at deployment.
2. After qat_quantizer.convert(), the saved state dict contains torchao
   custom quantized module tensors with no HF quantization_config. Neither
   AutoModelForImageTextToText.from_pretrained (this repo's own HF eval path)
   nor convert_hf_to_gguf.py can load it -- there is currently no way to
   evaluate or deploy what this script produces.

Fixing #1 properly likely means either finding/implementing a k-quant-shaped
fake-quantizer, or reframing this as its own independent PTQ-style int8/int4
point (evaluated via the HF runner, not exported to GGUF) rather than a QAT
comparison against the deployment quantizer. Fixing #2 means saving the
PRE-convert() fp16 weights (trained to be quantization-friendly) for GGUF
export, not the post-convert() torchao-native tensors -- and even then,
"evaluate via HF runner instead" only works if AutoModelForImageTextToText
can actually reload the post-convert() checkpoint, which needs a real
quantization_config wired up on the model config; unverified whether
transformers' torchao integration supports that for this quantizer without
hands-on GPU testing.

DECISION (given a scarce, unrepeatable GPU budget -- see spec §9's own
contingency: "QAT compute heavy -> PTQ carries the grid breadth"):
**do not run this for the first compression-grid pass.** Both issues are
real correctness risks, not cosmetic ones, and neither can be verified
without spending GPU time to find out -- exactly the kind of uncertain bet
a one-shot budget can't absorb. PTQ (quantize_ptq.py) and pruning (prune.py)
have no equivalent open questions and should be run first. Revisit QAT only
if the core grid succeeds with budget to spare, or document its omission
plainly in the writeup -- "attempted, found the fake-quantizer didn't match
the deployment quantizer, descoped to protect the core grid" is a more
honest and more informative research narrative than a rushed, possibly-
unusable result.
"""

from __future__ import annotations

import argparse
import json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--image-root", required=True, help="see train_sft.py --image-root")
    ap.add_argument("--policy", default="gold", choices=["gold", "teacher", "teacher-agree"])
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from torchao.quantization.qat import Int8DynActInt4WeightQATQuantizer
    from transformers import AutoModelForImageTextToText, AutoProcessor, Trainer, TrainingArguments

    from train_sft import build_messages, make_collator  # same data path as SFT — one contract

    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).parent.parent / "src"))
    from ledgerlens.schema import MAX_PIXELS, MIN_PIXELS

    records = [build_messages(json.loads(line), args.policy, image_root=args.image_root)
              for path in args.data for line in open(path)]

    # T4 (Turing, sm_75) has no bf16 support -- see train_sft.py for the same fix.
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(args.model, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)

    # Fake-quantize only the language model's linears; vision tower stays fp16.
    qat_quantizer = Int8DynActInt4WeightQATQuantizer(groupsize=128)
    model.model.language_model = qat_quantizer.prepare(model.model.language_model)

    # Full-parameter AdamW here would OOM a 16GB T4 (fp32 optimizer moments
    # alone are ~17GB for this model) and silently drifts from the spec's own
    # "LoRA-QAT to bound compute" plan. LoRA on top of the fake-quant modules
    # keeps this in the same memory class as train_sft.py's SFT run.
    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.enable_input_require_grads()

    collate = make_collator(processor)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.out + "-work", max_steps=args.steps, learning_rate=args.lr,
            per_device_train_batch_size=1, gradient_accumulation_steps=16,
            fp16=True, logging_steps=10, save_strategy="no", report_to=[],
            gradient_checkpointing=True, remove_unused_columns=False,
        ),
        train_dataset=Dataset.from_list(records),
        data_collator=collate,
    )
    trainer.train()

    # Convert fake-quant -> real quantized ops
    model.model.language_model = qat_quantizer.convert(model.model.language_model)
    model.save_pretrained(args.out)
    processor.save_pretrained(args.out)
    print(f"QAT int8-act/int4-weight -> {args.out}")


if __name__ == "__main__":
    main()
