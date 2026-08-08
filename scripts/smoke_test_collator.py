"""CPU-only, no-GPU-needed verification of the training data pipeline: real
message/target construction on real teacher-labeled data, through the real
Qwen2-VL processor and collator, checking actual tensor shapes and label
masking -- everything up to (not including) the model forward/backward pass.

This is not a substitute for --smoke-test in train_sft.py (which needs a GPU
to run the actual model), but it catches a large class of bugs (data format
errors, collator shape bugs, processor API drift across transformers
versions) for free, before ever spending GPU time. Run this first.

    uv run python scripts/smoke_test_collator.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from train_sft import STUDENT_DEFAULT, build_messages, make_collator  # noqa: E402


def main() -> None:
    data_file = Path(__file__).parent.parent / "data/distill/cord_train_teacher_0_200.jsonl"
    if not data_file.exists():
        sys.exit(f"expected real data at {data_file} -- run from repo root with data/distill/ present")

    records = [json.loads(line) for line in open(data_file)][:4]
    print(f"loaded {len(records)} real records from {data_file.name}")

    for policy in ("gold", "teacher", "teacher-agree"):
        messages = [build_messages(r, policy) for r in records]
        for m in messages:
            target = json.loads(m["messages"][1]["content"][0]["text"])
            assert set(target) == {"vendor", "date", "currency", "total", "tax", "line_items"}, \
                f"policy={policy}: target missing/extra keys: {target.keys()}"
        print(f"policy={policy}: build_messages OK on {len(messages)} records, target schema correct")

    print(f"\ndownloading processor for {STUDENT_DEFAULT} (small, config only, not model weights)...")
    from transformers import AutoProcessor
    from ledgerlens.schema import MAX_PIXELS, MIN_PIXELS

    processor = AutoProcessor.from_pretrained(STUDENT_DEFAULT, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
    print("processor loaded OK")

    # image_root: teacher_label.py stores paths relative to the train-data
    # dataset root (see its own comment) -- locally that's data/processed/,
    # which has the same images/cord_train/, images/sroie_train/ layout.
    image_root = Path(__file__).parent.parent / "data/processed"
    messages = [build_messages(r, "teacher-agree", image_root=image_root) for r in records]
    for m in messages:
        assert Path(m["images"][0]).exists(), f"missing image: {m['images'][0]}"
    collate = make_collator(processor)
    print("\nrunning the real collator on a batch of 4 real records...")
    batch = collate(messages)

    print(f"input_ids shape: {batch['input_ids'].shape}")
    print(f"labels shape: {batch['labels'].shape}")
    assert batch["input_ids"].shape == batch["labels"].shape

    n_masked = int((batch["labels"] == -100).sum())
    n_total = batch["labels"].numel()
    n_unmasked = n_total - n_masked
    print(f"labels: {n_masked}/{n_total} masked (-100), {n_unmasked} unmasked (real loss targets)")
    assert n_unmasked > 0, "BUG: every label is masked -- loss would be computed on nothing, silent no-op training"
    assert n_masked > 0, "BUG: nothing is masked -- loss would include the prompt/image tokens, not just the JSON answer"

    # Sanity: decode the unmasked span of row 0 and confirm it looks like our
    # target JSON, not prompt text or garbage.
    row0_unmasked = batch["input_ids"][0][batch["labels"][0] != -100]
    decoded = processor.tokenizer.decode(row0_unmasked)
    print(f"\nrow 0 unmasked span decodes to (should look like JSON):\n{decoded[:300]}")
    assert '"vendor"' in decoded or "vendor" in decoded, \
        f"BUG: unmasked span doesn't look like the JSON target: {decoded[:200]}"

    print("\n=== SMOKE TEST PASSED: real data -> real processor -> real collator, shapes and masking correct ===")


if __name__ == "__main__":
    main()
