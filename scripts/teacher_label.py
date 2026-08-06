"""Weekend 2, step 1: teacher (Qwen2.5-VL-7B, 4-bit) labels train images for distillation.

For each training sample the teacher generates an extraction. Outputs an SFT-ready
JSONL where each record carries BOTH the teacher output and the gold, plus an
agreement flag, so the blending policy (teacher-only vs gold-only vs agree-filtered)
stays a training-time choice — enabling the spec's ablation without re-labeling.

Run on the GPU box:
    uv run --extra gpu python scripts/teacher_label.py \
        --split data/processed/cord_train.jsonl --out data/distill/cord_train_teacher.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ledgerlens.data.unified import load_split  # noqa: E402
from ledgerlens.metrics import scalar_match  # noqa: E402
from ledgerlens.runners.base import get_runner  # noqa: E402
from ledgerlens.schema import PROMPT, parse_model_output  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    # BooleanOptionalAction so --no-load-in-4bit can actually override the
    # default (plain store_true+default=True can never be turned off from the CLI)
    ap.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--offset", type=int, default=0, help="skip the first N samples (chunking across sessions)")
    ap.add_argument("--limit", type=int, help="label at most N samples after --offset")
    args = ap.parse_args()

    samples = load_split(args.split)
    samples = samples[args.offset :]
    if args.limit:
        samples = samples[: args.limit]

    # Resume-safe: a rerun with the same --out (e.g. a Kaggle session getting
    # killed by the quota/time cap mid-chunk, then restarted) previously
    # opened with "w" and silently clobbered whatever had already been
    # labeled. Now skips ids already present in an existing output file and
    # appends the rest, so partial progress in a chunk is never lost.
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    already_done = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                already_done.add(json.loads(line)["id"])
        if already_done:
            print(f"resuming: {len(already_done)} samples already labeled in {out_path}", flush=True)
    samples = [s for s in samples if s.id not in already_done]

    runner = get_runner("hf", model_id=args.teacher, load_in_4bit=args.load_in_4bit)

    n_valid = n_agree = 0
    with open(out_path, "a") as f:
        for i, s in enumerate(samples):
            t0 = time.perf_counter()
            raw = runner.generate(s.image, PROMPT)
            pred = parse_model_output(raw)
            valid = pred is not None
            # agreement on the anchor field (total) — cheap proxy for label quality
            agrees = valid and scalar_match("total", pred.total, s.gold.total)
            n_valid += valid
            n_agree += agrees
            f.write(json.dumps({
                "id": s.id,
                "image": str(s.image),
                "prompt": PROMPT,
                "teacher_raw": raw,
                "teacher_json": pred.model_dump() if valid else None,
                "gold": s.gold.model_dump(),
                "teacher_valid": valid,
                "teacher_agrees_total": agrees,
                "latency_s": round(time.perf_counter() - t0, 3),
            }, ensure_ascii=False) + "\n")
            f.flush()
            if (i + 1) % 25 == 0:
                print(f"[{i + 1}/{len(samples)}] valid={n_valid} agree_total={n_agree}", flush=True)
    print(f"done: {len(samples)} samples, {n_valid} valid JSON, {n_agree} agree on total -> {out_path}")


if __name__ == "__main__":
    main()
