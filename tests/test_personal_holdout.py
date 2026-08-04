import json

from PIL import Image

from ledgerlens.data.personal import convert_personal
from ledgerlens.data.unified import load_split


def _make_raw(tmp_path, names):
    raw = tmp_path / "personal_holdout" / "raw"
    raw.mkdir(parents=True)
    for name in names:
        Image.new("RGB", (10, 10), "white").save(raw / name)
    return raw


def test_first_run_scaffolds_blank_labels(tmp_path):
    _make_raw(tmp_path, ["r1.jpg", "r2.jpg"])
    convert_personal(tmp_path)
    labels = tmp_path / "personal_holdout" / "labels"
    assert (labels / "r1.json").exists()
    assert (labels / "r2.json").exists()
    blank = json.loads((labels / "r1.json").read_text())
    assert blank == {"vendor": None, "date": None, "currency": None,
                     "total": None, "tax": None, "line_items": []}
    # no split written yet -- nothing has been filled in
    assert not (tmp_path / "personal_holdout.jsonl").exists()


def test_filled_labels_produce_split_blank_ones_skipped(tmp_path):
    _make_raw(tmp_path, ["r1.jpg", "r2.jpg"])
    convert_personal(tmp_path)  # scaffold
    labels = tmp_path / "personal_holdout" / "labels"
    (labels / "r1.json").write_text(json.dumps({
        "vendor": "Real Shop", "date": "2024-05-01", "currency": "USD",
        "total": 12.5, "tax": 1.0, "line_items": [],
    }))
    # r2.json left blank

    convert_personal(tmp_path)  # second pass: build the split

    samples = load_split(tmp_path / "personal_holdout.jsonl")
    assert len(samples) == 1
    assert samples[0].id == "personal_r1"
    assert samples[0].source == "personal"
    assert samples[0].gold.vendor == "Real Shop"
    assert samples[0].image.exists()


def test_bad_field_name_fails_loudly(tmp_path):
    _make_raw(tmp_path, ["r1.jpg"])
    convert_personal(tmp_path)
    labels = tmp_path / "personal_holdout" / "labels"
    (labels / "r1.json").write_text(json.dumps({
        "vendour": "Typo'd Key", "total": 5.0,  # typo'd field name
    }))
    try:
        convert_personal(tmp_path)
        assert False, "expected a validation error"
    except Exception:
        pass


def test_no_raw_dir_raises_clear_error(tmp_path):
    try:
        convert_personal(tmp_path)
        assert False, "expected SystemExit"
    except SystemExit as e:
        assert "Photograph" in str(e)
