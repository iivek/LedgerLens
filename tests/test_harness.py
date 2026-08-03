import csv
import json

from PIL import Image

from ledgerlens.data.unified import load_split, write_split
from ledgerlens.harness import run_eval
from ledgerlens.runners.mock import MockRunner

GOLDS = [
    {"vendor": "ACME Mart", "date": "2024-03-05", "currency": "USD", "total": 19.0,
     "tax": None, "line_items": [{"description": "Coffee", "qty": 2, "unit_price": 3.5, "total": 7.0}]},
    {"vendor": "Beta Store", "date": "2023-11-20", "currency": "EUR", "total": 100.0,
     "tax": 20.0, "line_items": []},
]


def _make_split(tmp_path):
    (tmp_path / "images").mkdir()
    records = []
    for i, gold in enumerate(GOLDS):
        img = tmp_path / "images" / f"s{i}.png"
        Image.new("RGB", (10, 10), "white").save(img)
        records.append({"id": f"s{i}", "image": f"images/s{i}.png", "source": "test", "gold": gold})
    path = tmp_path / "split.jsonl"
    write_split(records, path)
    return path


def test_end_to_end_perfect_mock(tmp_path):
    split = _make_split(tmp_path)
    samples = load_split(split)
    runner = MockRunner(gold_by_image={str(s.image): s.gold.model_dump() for s in samples})
    row = run_eval(runner, split, tag="mock-perfect", host="test",
                   runs_root=tmp_path / "runs", results_csv=tmp_path / "results.csv")
    assert float(row["json_validity_rate"]) == 1.0
    assert float(row["fields_em_mean"]) == 1.0
    assert float(row["line_item_f1"]) == 1.0
    # per-sample predictions persisted for audit
    preds = [json.loads(l) for l in (tmp_path / "runs/mock-perfect/predictions.jsonl").read_text().splitlines()]
    assert len(preds) == 2 and preds[0]["raw"]


def test_results_csv_accumulates_rows(tmp_path):
    split = _make_split(tmp_path)
    samples = load_split(split)
    gold_map = {str(s.image): s.gold.model_dump() for s in samples}
    csv_path = tmp_path / "results.csv"
    run_eval(MockRunner(gold_by_image=gold_map), split, tag="a", host="test",
             runs_root=tmp_path / "runs", results_csv=csv_path)
    run_eval(MockRunner(gold_by_image=gold_map, invalid_json_rate=1.0), split, tag="b",
             host="test", runs_root=tmp_path / "runs", results_csv=csv_path)
    rows = list(csv.DictReader(open(csv_path)))
    assert [r["tag"] for r in rows] == ["a", "b"]
    assert float(rows[1]["json_validity_rate"]) == 0.0


def test_runner_exception_counts_as_invalid(tmp_path):
    split = _make_split(tmp_path)

    class Boom:
        name = "boom"
        def generate(self, image_path, prompt):
            raise RuntimeError("model crashed")

    row = run_eval(Boom(), split, tag="boom", host="test",
                   runs_root=tmp_path / "runs", results_csv=tmp_path / "results.csv")
    assert float(row["json_validity_rate"]) == 0.0
