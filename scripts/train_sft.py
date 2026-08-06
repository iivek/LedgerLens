"""Weekend 2, step 2: LoRA SFT of the 2B student on teacher outputs blended with gold.

Targets are built FIELD-BY-FIELD, not by picking one whole source per record:

  - A field the sample's source annotates (see metrics.ANNOTATED_FIELDS) always
    uses gold for that field, when policy="gold" or the field-level agreement
    check applies -- verified ground truth is never overridden by an unverified
    guess.
  - A field the source does NOT annotate (e.g. vendor/date/currency on CORD,
    line_items/tax on SROIE) always uses the teacher's value, regardless of
    policy. Gold is null there for every sample of that source -- supervising
    on it would teach the model to output null for a field the receipt visibly
    has, which is not a "no distillation" baseline, it's actively wrong
    supervision. This was a real bug (caught by two independent reviews) in an
    earlier whole-record gold/teacher/teacher-agree design.

Policy controls what happens on ANNOTATED fields when teacher and gold might
disagree (the actual ablation question -- see spec's "ablation of teacher-data
vs ground-truth-only"):
  gold          — always trust gold on annotated fields (no distillation there)
  teacher       — trust the teacher's read whenever it produced one
  teacher-agree — trust the teacher only where it agrees with gold on THAT
                  field specifically (line_items: Hungarian F1 above a
                  threshold). Per-field, not just a whole-record `total` proxy
                  -- an agreeing total does not guarantee vendor or line_items
                  are also correct, which the old total-only check missed.

    uv run --extra gpu python scripts/train_sft.py \
        --data data/distill/cord_train_teacher_*.jsonl data/distill/sroie_train_teacher_*.jsonl \
        --policy teacher-agree --out runs/sft-2b-teacher-agree --merge
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ledgerlens.metrics import annotated_fields, line_item_prf, scalar_match  # noqa: E402
from ledgerlens.schema import SCALAR_FIELDS, LineItem  # noqa: E402

STUDENT_DEFAULT = "Qwen/Qwen2-VL-2B-Instruct"
LINE_ITEM_AGREE_F1 = 0.7  # threshold for "teacher's line items agree with gold" in teacher-agree


def _line_items_agree(teacher_items: list, gold_items: list) -> bool:
    t = [LineItem.model_validate(i) for i in (teacher_items or [])]
    g = [LineItem.model_validate(i) for i in (gold_items or [])]
    _, _, f1 = line_item_prf(t, g)
    return f1 >= LINE_ITEM_AGREE_F1


def _field_agrees(field: str, teacher_value, gold_value) -> bool:
    if field == "line_items":
        return _line_items_agree(teacher_value, gold_value)
    return scalar_match(field, teacher_value, gold_value)


def build_target(rec: dict, policy: str) -> dict:
    """Field-level blend. rec: one teacher_label.py output record."""
    source = rec["source"]
    gold = rec["gold"]
    teacher = rec.get("teacher_json")
    annotated = annotated_fields(source)

    target = {}
    for field in (*SCALAR_FIELDS, "line_items"):
        gold_value = gold.get(field)
        teacher_value = teacher.get(field) if teacher else None

        if field not in annotated:
            # No ground truth exists for this field on this source at all --
            # the teacher's guess (possibly None if it produced invalid JSON)
            # is the only signal available, for every policy.
            target[field] = teacher_value
            continue

        if policy == "gold" or teacher is None:
            target[field] = gold_value
        elif policy == "teacher":
            target[field] = teacher_value if teacher_value is not None else gold_value
        elif policy == "teacher-agree":
            target[field] = teacher_value if _field_agrees(field, teacher_value, gold_value) else gold_value
        else:
            raise ValueError(policy)
    return target


def build_messages(rec: dict, policy: str) -> dict:
    target = build_target(rec, policy)
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


def make_collator(processor, max_length: int | None = None):
    """Batch collator shared by SFT and QAT: loss only on the assistant JSON turn.

    No truncation is applied to the processor call: HF's default right-side
    truncation would silently cut off the assistant's JSON answer (the training
    target) rather than the prompt, corrupting labels without any error. Instead,
    an overlong batch raises loudly with the offending sample ids so it can be
    fixed in the data rather than discovered as an unexplained accuracy drop.
    """
    from qwen_vl_utils import process_vision_info

    limit = max_length or getattr(processor.tokenizer, "model_max_length", None)

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
        if limit and inputs["input_ids"].shape[1] > limit:
            offenders = [ex["images"][0] for ex in batch]
            raise ValueError(
                f"batch sequence length {inputs['input_ids'].shape[1]} exceeds "
                f"model_max_length={limit}; offending samples: {offenders}. "
                "Fix by trimming these samples' line items or raising max_pixels "
                "cap, not by truncating — truncation would cut the label."
            )
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

    records = [build_messages(json.loads(line), args.policy)
              for path in args.data for line in open(path)]
    print(f"{len(records)} training samples (policy={args.policy})")

    processor = AutoProcessor.from_pretrained(
        args.student, min_pixels=256 * 28 * 28, max_pixels=1024 * 28 * 28
    )
    # T4 (Turing, sm_75) has no bf16 support -- TrainingArguments hard-fails at
    # init and model loading silently mis-casts under bf16 on this hardware.
    # fp16 needs the usual loss-scaling care with LoRA, which Trainer handles
    # via GradScaler when bf16=False, fp16=True.
    model = AutoModelForImageTextToText.from_pretrained(
        args.student, torch_dtype=torch.float16, device_map="auto"
    )
    lora = LoraConfig(
        r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    # LoRA freezes the base model's input embeddings, so gradient checkpointing
    # (enabled below) sees no tensor requiring grad at the graph's entry point
    # and errors ("element 0 of tensors does not require grad") without this.
    model.enable_input_require_grads()

    collate = make_collator(processor)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.out, num_train_epochs=args.epochs, learning_rate=args.lr,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            fp16=True, logging_steps=10, save_strategy="epoch", report_to=[],
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
