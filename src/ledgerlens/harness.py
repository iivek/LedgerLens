"""Eval harness: dataset split + runner -> per-sample JSONL + one aggregate CSV row.

Design rules (spec §5):
- Fixed metrics, fixed prompt, greedy decoding. Every model variant is evaluated
  by this exact code path, so rows are comparable points on the Pareto grid.
- Per-sample raw outputs are persisted (runs/<tag>/predictions.jsonl) so any
  aggregate number can be audited or re-scored without re-running the model.
- Aggregates append to a single results CSV; one row = one (variant, split, host).
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict
from pathlib import Path

from .data.unified import Sample, load_split
from .metrics import SampleScore, aggregate, score_sample
from .runners.base import ModelRunner
from .schema import PROMPT, parse_model_output

AGGREGATE_COLUMNS_FIRST = ["tag", "runner", "split", "host", "timestamp"]


def evaluate(runner: ModelRunner, samples: list[Sample], run_dir: Path,
             prompt: str = PROMPT, progress: bool = True) -> list[SampleScore]:
    run_dir.mkdir(parents=True, exist_ok=True)
    pred_path = run_dir / "predictions.jsonl"
    scores: list[SampleScore] = []
    with open(pred_path, "w") as f:
        for i, s in enumerate(samples):
            t0 = time.perf_counter()
            try:
                raw = runner.generate(s.image, prompt)
                error = None
            except Exception as e:  # runner failure = invalid output, not a crash
                raw, error = "", f"{type(e).__name__}: {e}"
            latency = time.perf_counter() - t0
            pred = parse_model_output(raw)
            score = score_sample(s.id, pred, s.gold, latency_s=latency, source=s.source)
            scores.append(score)
            f.write(json.dumps({
                "id": s.id, "source": s.source, "raw": raw, "error": error,
                "latency_s": latency, "score": asdict(score),
            }, ensure_ascii=False) + "\n")
            if progress and (i + 1) % 10 == 0:
                print(f"  [{i + 1}/{len(samples)}] {s.id}")
    return scores


def append_results_csv(csv_path: Path, row: dict) -> None:
    """Append one aggregate row, growing the header union if new columns appear."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    existing_rows: list[dict] = []
    if csv_path.exists():
        with open(csv_path, newline="") as f:
            existing_rows = list(csv.DictReader(f))
    all_rows = existing_rows + [row]
    cols = list(dict.fromkeys(
        AGGREGATE_COLUMNS_FIRST
        + [c for r in all_rows for c in r.keys() if c not in AGGREGATE_COLUMNS_FIRST]
    ))
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(all_rows)


def run_eval(runner: ModelRunner, split_path: Path, tag: str, host: str,
             runs_root: Path = Path("runs"), results_csv: Path = Path("results/results.csv"),
             limit: int | None = None) -> dict:
    samples = load_split(split_path)
    if limit:
        samples = samples[:limit]
    run_dir = runs_root / tag
    print(f"Evaluating {runner.name} on {split_path} ({len(samples)} samples) -> {run_dir}")
    scores = evaluate(runner, samples, run_dir)
    agg = aggregate(scores)
    row = {
        "tag": tag, "runner": runner.name, "split": str(split_path), "host": host,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        **{k: f"{v:.4f}" if isinstance(v, float) else v for k, v in agg.items()},
    }
    append_results_csv(results_csv, row)
    (run_dir / "aggregate.json").write_text(json.dumps(row, indent=2))
    return row
