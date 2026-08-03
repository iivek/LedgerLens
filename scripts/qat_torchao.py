"""Weekend 3: QAT via torchao on the LoRA-merged student (int8 dyn-act / int4 weight).

Strategy per spec: bound compute by running QAT as a *short* fine-tune with fake
quantization on the LLM linear layers, reusing the SFT data pipeline. After
convergence, the fake-quant modules convert to real quantized ops; the resulting
state dict also serves as a better initialization for GGUF int4 export.

    uv run --extra gpu python scripts/qat_torchao.py \
        --model runs/sft-2b-teacher-agree-merged \
        --data data/distill/cord_train_teacher.jsonl --policy gold \
        --steps 500 --out models/student2b-qat-int4
"""

from __future__ import annotations

import argparse
import json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--policy", default="gold", choices=["gold", "teacher", "teacher-agree"])
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    from datasets import Dataset
    from torchao.quantization.qat import Int8DynActInt4WeightQATQuantizer
    from transformers import AutoModelForImageTextToText, AutoProcessor, Trainer, TrainingArguments

    from train_sft import build_messages, make_collator  # same data path as SFT — one contract

    records = []
    for path in args.data:
        with open(path) as f:
            records += [m for m in (build_messages(json.loads(l), args.policy) for l in f) if m]

    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(args.model)

    # Fake-quantize only the language model's linears; vision tower stays fp16.
    qat_quantizer = Int8DynActInt4WeightQATQuantizer(groupsize=128)
    model.model.language_model = qat_quantizer.prepare(model.model.language_model)

    collate = make_collator(processor)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.out + "-work", max_steps=args.steps, learning_rate=args.lr,
            per_device_train_batch_size=1, gradient_accumulation_steps=16,
            bf16=True, logging_steps=10, save_strategy="no", report_to=[],
            remove_unused_columns=False,
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
