import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from ledgerlens.data.unified import write_split
from rescore import vendor_disjoint_ids  # noqa: E402


def _rec(id_, vendor):
    return {"id": id_, "image": "images/x.png", "source": "sroie",
            "gold": {"vendor": vendor, "date": None, "currency": None, "total": 1.0, "tax": None, "line_items": []}}


def test_vendor_disjoint_excludes_overlapping_vendors(tmp_path):
    (tmp_path / "images").mkdir()
    from PIL import Image
    Image.new("RGB", (4, 4)).save(tmp_path / "images" / "x.png")

    train = tmp_path / "train.jsonl"
    test = tmp_path / "test.jsonl"
    write_split([_rec("t1", "ACME Mart"), _rec("t2", "Beta Store")], train)
    write_split([
        _rec("s1", "ACME Mart"),        # overlaps train -> excluded
        _rec("s2", "acme mart"),        # case-insensitive overlap -> excluded
        _rec("s3", "Totally New Shop"), # disjoint -> kept
        _rec("s4", None),               # no vendor at all -> kept (can't overlap)
    ], test)

    kept = vendor_disjoint_ids(str(train), str(test))
    assert kept == {"s3", "s4"}
