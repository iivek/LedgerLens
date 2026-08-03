"""Weekend 2, step 2: LoRA SFT of the 2B student on teacher outputs blended with gold.

Blend policies (the spec's ablation axis):
  gold          — ground truth targets only (no distillation baseline)
  teacher       — teacher JSON whenever valid, else gold
  teacher-agree — teacher JSON only when it agrees with gold on total, else gold

Targets are always serialized from structured JSON (compact, key-ordered), so the
student learns the exact output contract the harness scores.

    uv run --extra gpu python scripts/train_sft.py \
        --data data/distill/cord_train_teacher.jsonl data/distill/synth_train_teacher.jsonl \
        --policy teacher-agree --out runs/sft-2b-teacher-agree --merge
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

STUDENT_DEFAULT = "Qwen/Qwen2-VL-2B-Instruct"


def build_messages(rec: dict, policy: str) -> dict | None:
    if policy == "gold":
        target = rec["gold"]
    elif policy == "teacher":
        target = rec["teacher_json"] or rec["gold"]
    elif policy == "teacher-agree":
        target = rec["teacher_json"] if rec.get("teacher_agrees_total") else rec["gold"]
    else:
        raise ValueError(policy)
    if target is None:
        return None
    return {
        "images": [rec["image"]],
        "messages": [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": rec["prompt"]},
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": json.dumps(target, ensure_ascii=False)},
            ]},
        ],
    }


def make_collator(processor):
    """Batch collator shared by SFT and QAT: loss only on the assistant JSON turn."""
    from qwen_vl_utils import process_vision_info

    def collate(batch):
        texts, images = [], []
        for ex in batch:
            msgs = [{"role": m["role"], "content": [
                dict(c, image=f"file://{ex['images'][0]}") if c["type"] == "image" else c
                for c in m["content"]
            ]} for m in ex["messages"]]
            texts.append(processor.apply_chat_template(msgs, tokenize=False))
            imgs, _ = process_vision_info(msgs)
            images.append(imgs)
        inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
        labels = inputs["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        # mask everything before the assistant turn so loss covers only the JSON answer
        im_start = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
        for row in range(labels.shape[0]):
            starts = (inputs["input_ids"][row] == im_start).nonzero().flatten()
            if len(starts) >= 2:
                labels[row, : starts[-1] + 1] = -100
        inputs["labels"] = labels
        return inputs

    return collate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", default=STUDENT_DEFAULT)
    ap.add_argument("--data", nargs="+", required=True, help="teacher_label.py output jsonl(s)")
    ap.add_argument("--policy", choices=["gold", "teacher", "teacher-agree"], default="teacher-agree")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--merge", action="store_true", help="merge LoRA into base weights after training")
    args = ap.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForImageTextToText, AutoProcessor, Trainer,
                              TrainingArguments)

    records = []
    for path in args.data:
        with open(path) as f:
            for line in f:
                m = build_messages(json.loads(line), args.policy)
                if m:
                    records.append(m)
    print(f"{len(records)} training samples (policy={args.policy})")

    processor = AutoProcessor.from_pretrained(
        args.student, min_pixels=256 * 28 * 28, max_pixels=1024 * 28 * 28
    )
    model = AutoModelForImageTextToText.from_pretrained(
        args.student, torch_dtype=torch.bfloat16, device_map="auto"
    )
    lora = LoraConfig(
        r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    collate = make_collator(processor)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.out, num_train_epochs=args.epochs, learning_rate=args.lr,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            bf16=True, logging_steps=10, save_strategy="epoch", report_to=[],
            gradient_checkpointing=True, remove_unused_columns=False,
        ),
        train_dataset=Dataset.from_list(records),
        data_collator=collate,
    )
    trainer.train()
    trainer.save_model(args.out)
    processor.save_pretrained(args.out)

    if args.merge:
        merged_dir = Path(args.out).with_name(Path(args.out).name + "-merged")
        merged = model.merge_and_unload()
        merged.save_pretrained(merged_dir)
        processor.save_pretrained(merged_dir)
        print(f"merged model -> {merged_dir}")


if __name__ == "__main__":
    main()
