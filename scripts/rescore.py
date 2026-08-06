"""Rescore a saved run's predictions.jsonl against the current metrics code and
current gold, without calling the model again.

Exists because the harness persists raw model output specifically so a metric
bug found later doesn't require burning compute to fix: this replays those
raw outputs through score_sample()/aggregate() and appends a corrected row to
the results CSV under a new tag, so both the original and corrected numbers
stay in the audit trail.

    uv run python scripts/rescore.py --run runs/stock2b-q4km-cord \
        --split data/processed/cord_test.jsonl --tag stock2b-q4km-cord-rescored
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ledgerlens.data.unified import load_split  # noqa: E402
from ledgerlens.harness import append_results_csv  # noqa: E402
from ledgerlens.metrics import aggregate, score_sample  # noqa: E402
from ledgerlens.schema import parse_model_output  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="existing runs/<tag> directory")
    ap.add_argument("--split", required=True, help="unified jsonl split (current gold)")
    ap.add_argument("--tag", required=True, help="new tag for the corrected row")
    ap.add_argument("--host", default=None, help="defaults to the original row's host if found")
    ap.add_argument("--results", default="results/results.csv")
    args = ap.parse_args()

    gold_by_id = {s.id: s.gold for s in load_split(args.split)}
    pred_path = Path(args.run) / "predictions.jsonl"

    scores = []
    with open(pred_path) as f:
        for line in f:
            rec = json.loads(line)
            gold = gold_by_id.get(rec["id"])
            if gold is None:
                raise SystemExit(f"{rec['id']} not found in {args.split} -- wrong split?")
            pred = parse_model_output(rec["raw"])
            scores.append(score_sample(rec["id"], pred, gold, latency_s=rec["latency_s"], source=rec["source"]))

    agg = aggregate(scores)
    host = args.host or "rescored"
    row = {
        "tag": args.tag, "runner": args.tag, "split": args.split, "host": host,
        "timestamp": __import__("time").strftime("%Y-%m-%dT%H:%M:%S"),
        **{k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in agg.items()},
    }
    append_results_csv(Path(args.results), row)
    width = max(len(k) for k in row)
    for k, v in row.items():
        print(f"{k:<{width}}  {v}")


if __name__ == "__main__":
    main()
