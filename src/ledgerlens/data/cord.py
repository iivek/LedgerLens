"""CORD-v2 -> unified format.

Source: naver-clova-ix/cord-v2 on HuggingFace (research use; see repo license note).
CORD annotates menu line items and totals but NOT vendor/date/currency — those are
stored as null and score as trivially-correct only when the model also outputs null
(the spec's both-absent rule). CORD therefore mainly exercises line items + totals;
SROIE covers vendor/date.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..normalize import normalize_number
from .unified import write_split

HF_ID = "naver-clova-ix/cord-v2"


def _as_list(x) -> list:
    if x is None:
        return []
    return x if isinstance(x, list) else [x]


def _join_text(x) -> str | None:
    """CORD splits one item's name across multiple annotated text lines as a
    list (e.g. ["-", "Gyro Platter Large"] or ["TAKEOUT(1):", "Viet Latte"]).
    Taking only the first element (the original approach) frequently picked a
    junk prefix line over the actual name -- verified against real CORD data,
    not assumed. Joining preserves the real name at the cost of occasionally
    including a junk prefix, which is strictly better than silently dropping
    or replacing the name.
    """
    if x is None:
        return None
    if isinstance(x, list):
        parts = [_join_text(v) for v in x]
        parts = [p for p in parts if p]
        return " ".join(parts) if parts else None
    return str(x) if x != "" else None


def _first_valid_number(x) -> float | None:
    """CORD numeric fields occasionally contain junk mixed with the real
    value (e.g. cnt: ["TRIPPLE CHEESE", "1"]) or multiple OCR readings of the
    same number (e.g. total_price: ["55,834", "55,800"]). Taking the first
    raw element (the original approach) could pick unparseable junk over a
    valid number anywhere else in the list. This tries each element in order
    and returns the first that actually parses as a number, which handles
    both cases without assuming position matters.
    """
    for v in _as_list(x):
        n = normalize_number(v)
        if n is not None:
            return n
    return None


def _gt_to_extraction(gt_parse: dict) -> dict:
    line_items = []
    for item in _as_list(gt_parse.get("menu")):
        if not isinstance(item, dict):
            continue
        line_items.append({
            "description": _join_text(item.get("nm")),
            "qty": _first_valid_number(item.get("cnt")),
            "unit_price": _first_valid_number(item.get("unitprice")),
            "total": _first_valid_number(item.get("price")),
        })
    total_block = gt_parse.get("total") or {}
    sub_block = gt_parse.get("sub_total") or {}
    if isinstance(total_block, list):  # occasionally a list of blocks
        total_block = total_block[0] if total_block else {}
    if isinstance(sub_block, list):
        sub_block = sub_block[0] if sub_block else {}
    return {
        "vendor": None, "date": None, "currency": None,
        "total": _first_valid_number(total_block.get("total_price")),
        "tax": _first_valid_number(sub_block.get("tax_price")),
        "line_items": line_items,
    }


def convert_cord(out_dir: Path) -> None:
    from datasets import load_dataset  # requires the `data` extra

    for split, out_name in (("train", "cord_train"), ("validation", "cord_val"), ("test", "cord_test")):
        ds = load_dataset(HF_ID, split=split)
        img_dir = out_dir / "images" / out_name
        img_dir.mkdir(parents=True, exist_ok=True)
        records = []
        for i, ex in enumerate(ds):
            sample_id = f"{out_name}_{i:04d}"
            img_path = img_dir / f"{sample_id}.jpg"
            if not img_path.exists():
                ex["image"].convert("RGB").save(img_path, quality=92)
            gt_parse = json.loads(ex["ground_truth"])["gt_parse"]
            records.append({
                "id": sample_id,
                "image": str(img_path.relative_to(out_dir)),
                "source": "cord",
                "gold": _gt_to_extraction(gt_parse),
            })
        write_split(records, out_dir / f"{out_name}.jsonl")
        print(f"{out_name}: {len(records)} samples -> {out_dir / (out_name + '.jsonl')}")
