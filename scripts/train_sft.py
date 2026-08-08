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

ALWAYS run --smoke-test first on new data/hardware -- it exercises the full
data/model/collator/save path in a couple minutes instead of hours. There is
no budget for a failed full run.

    uv run --extra gpu python scripts/train_sft.py \
        --data data/distill/cord_train_teacher_*.jsonl data/distill/sroie_train_teacher_*.jsonl \
        --image-root /kaggle/input/datasets/<account>/ledgerlens-train-data \
        --eval-data data/distill/cord_val_teacher.jsonl --eval-image-root <val-data-mount> \
        --policy teacher-agree --out runs/sft-2b-teacher-agree --smoke-test   # verify first

    uv run --extra gpu python scripts/train_sft.py \
        --data data/distill/cord_train_teacher_*.jsonl data/distill/sroie_train_teacher_*.jsonl \
        --image-root /kaggle/input/datasets/<account>/ledgerlens-train-data \
        --eval-data data/distill/cord_val_teacher.jsonl --eval-image-root <val-data-mount> \
        --policy teacher-agree --out runs/sft-2b-teacher-agree --merge   # the real run

Re-running the exact same command (same --out) after an interrupted session
resumes from the latest checkpoint automatically -- see the --out checkpoint
detection below.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ledgerlens.metrics import annotated_fields, line_item_prf, scalar_match  # noqa: E402
from ledgerlens.schema import MAX_PIXELS, MIN_PIXELS, SCALAR_FIELDS, LineItem  # noqa: E402

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


def build_messages(rec: dict, policy: str, image_root: str | Path | None = None) -> dict:
    """image_root: directory rec["image"] (a relative path, per teacher_label.py's
    format) is resolved against. Required in practice -- omitting it only works
    if the caller already put an absolute/resolvable path in rec["image"].
    """
    target = build_target(rec, policy)
    image_path = str(Path(image_root) / rec["image"]) if image_root else rec["image"]
    return {
        "images": [image_path],
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
    ap.add_argument("--image-root", required=True,
                    help="directory each --data record's relative image path resolves against "
                         "-- the mounted train-data dataset root (e.g. the directory containing "
                         "cord_train.jsonl/images/ on this Kaggle session), NOT the labeled "
                         "jsonl's own directory, which is a different mount/location on Kaggle")
    ap.add_argument("--policy", choices=["gold", "teacher", "teacher-agree"], default="teacher-agree")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--merge", action="store_true", help="merge LoRA into base weights after training")
    ap.add_argument("--eval-data", nargs="*", default=None,
                    help="teacher_label.py-format jsonl(s) (or gold-only records -- teacher_json "
                         "may be absent) held out for eval_loss-based checkpoint selection, e.g. "
                         "a cord_val/sroie_val split. Optional but strongly recommended: without "
                         "it, save_strategy falls back to 'epoch' with no way to tell whether a "
                         "later checkpoint is actually better or just more overfit. NOTE: cord_val "
                         "was never uploaded to the ledgerlens-train-data Kaggle dataset (only "
                         "*_train splits were) -- it must be added to whatever gets mounted this "
                         "session before --eval-data can point at it.")
    ap.add_argument("--eval-image-root", default=None,
                    help="image root for --eval-data records, if different from --image-root "
                         "(e.g. eval data mounted from a separate dataset). Defaults to --image-root.")
    ap.add_argument("--save-steps", type=int, default=50,
                    help="checkpoint frequency when --eval-data is given (also the resume "
                         "granularity -- a session killed mid-epoch loses at most this many "
                         "steps, not the whole epoch)")
    ap.add_argument("--smoke-test", action="store_true",
                    help="run on --smoke-test-n samples for --smoke-test-n steps and exit before "
                         "the real training loop -- verifies the full data/model/collator/save "
                         "path end-to-end in minutes, not hours. ALWAYS run this before the real "
                         "job on new data/hardware; there is no budget for a failed full run.")
    ap.add_argument("--smoke-test-n", type=int, default=4)
    args = ap.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForImageTextToText, AutoProcessor, Trainer,
                              TrainingArguments)

    def load_records(paths, policy, image_root, limit=None):
        recs = [build_messages(json.loads(line), policy, image_root=image_root)
               for path in paths for line in open(path)]
        return recs[:limit] if limit else recs

    train_limit = args.smoke_test_n if args.smoke_test else None
    if args.smoke_test:
        print(f"=== SMOKE TEST: {args.smoke_test_n} samples, 1 step, no save/merge ===")

    records = load_records(args.data, args.policy, args.image_root, limit=train_limit)
    print(f"{len(records)} training samples (policy={args.policy})")
    missing = [m["images"][0] for m in records if not Path(m["images"][0]).exists()][:5]
    if missing:
        raise SystemExit(f"--image-root {args.image_root} looks wrong -- sample image(s) not "
                         f"found, e.g.: {missing}")

    eval_records = None
    if args.eval_data and not args.smoke_test:
        eval_image_root = args.eval_image_root or args.image_root
        eval_records = load_records(args.eval_data, args.policy, eval_image_root)
        print(f"{len(eval_records)} eval samples")

    # Matches the eval-time pixel budget (hf_vlm.py/llamacpp deployment) so
    # training and evaluation see the same effective image resolution --
    # previously this used a smaller max_pixels (1024*28*28 vs eval's
    # 1280*28*28) with no documented reason, an unacknowledged train/eval
    # resolution mismatch. If this OOMs on the T4, lower it here deliberately
    # (and note it in results.csv's host/config) rather than let it silently
    # drift from what evaluation actually measures.
    processor = AutoProcessor.from_pretrained(
        args.student, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS
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

    if args.smoke_test:
        # 1-2 real optimizer steps on a handful of samples: exercises data
        # loading, message/target construction, the collator's shape and
        # length-guard logic, fp16 + LoRA + gradient-checkpointing
        # compatibility, and a full forward/backward/save path -- everything
        # that could crash a multi-hour run -- in minutes. No save, no merge:
        # this checkpoint is throwaway, only the "did it crash" signal matters.
        smoke_args = TrainingArguments(
            output_dir=args.out + "-smoke", max_steps=min(2, max(1, len(records))),
            per_device_train_batch_size=1, gradient_accumulation_steps=1,
            fp16=True, logging_steps=1, save_strategy="no", report_to=[],
            gradient_checkpointing=True, remove_unused_columns=False,
        )
        Trainer(model=model, args=smoke_args, train_dataset=Dataset.from_list(records),
               data_collator=collate).train()
        print("=== SMOKE TEST PASSED: forward/backward/optimizer step completed without error ===")
        print("=== Re-run without --smoke-test for the real job ===")
        return

    # Resume support: Kaggle sessions can be killed mid-run, and re-running
    # this exact command with the same --out is meant to pick up where it
    # left off rather than restart from scratch -- there's no budget for a
    # second full run if a long one gets cut off partway through.
    resume = None
    if Path(args.out).is_dir() and any(Path(args.out).glob("checkpoint-*")):
        resume = True
        print(f"=== resuming from latest checkpoint in {args.out} ===")

    training_kwargs = dict(
        output_dir=args.out, num_train_epochs=args.epochs, learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        fp16=True, logging_steps=10, report_to=[],
        gradient_checkpointing=True, remove_unused_columns=False,
    )
    eval_dataset = None
    if eval_records is not None:
        # Frequent step-based checkpoints (not just per-epoch) so a killed
        # session loses at most --save-steps of progress, and eval_loss picks
        # the best one seen rather than trusting "whatever finished last" --
        # LoRA on a few thousand samples can start overfitting well within
        # a single epoch.
        eval_dataset = Dataset.from_list(eval_records)
        training_kwargs.update(
            eval_strategy="steps", eval_steps=args.save_steps,
            save_strategy="steps", save_steps=args.save_steps,
            load_best_model_at_end=True, metric_for_best_model="eval_loss",
            save_total_limit=3,
        )
    else:
        training_kwargs["save_strategy"] = "epoch"
        print("=== WARNING: no --eval-data given; saving on epoch boundaries with no "
              "eval-based selection. Strongly prefer passing --eval-data. ===")

    trainer = Trainer(
        model=model,
        args=TrainingArguments(**training_kwargs),
        train_dataset=Dataset.from_list(records),
        eval_dataset=eval_dataset,
        data_collator=collate,
    )
    trainer.train(resume_from_checkpoint=resume)
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
