"""SROIE 2019 (task 3 key-value annotations) -> unified format.

Expected input layout is the widely mirrored "SROIE2019" folder structure
(e.g. Kaggle dataset urbikn/sroie-datasetv2), passed via --raw or placed at
data/raw/SROIE2019:

    SROIE2019/
      train/{img,entities}/   # X00016469612.jpg + X00016469612.txt (JSON)
      test/{img,entities}/

Entities files: {"company": ..., "date": ..., "address": ..., "total": ...}.
SROIE has no line-item or tax annotations -> null; company maps to vendor.
Research-use license; see repo license note.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from ..normalize import normalize_date, normalize_number
from .unified import write_split

DEFAULT_RAW = Path("data/raw/SROIE2019")


def convert_sroie(out_dir: Path, raw_dir: Path = DEFAULT_RAW) -> None:
    if not raw_dir.exists():
        raise SystemExit(
            f"SROIE raw data not found at {raw_dir}.\n"
            "Download the 'SROIE2019' folder (e.g. Kaggle: urbikn/sroie-datasetv2) "
            f"and place it there, or pass a different path."
        )
    for split, out_name in (("train", "sroie_train"), ("test", "sroie_test")):
        ent_dir = raw_dir / split / "entities"
        img_src_dir = raw_dir / split / "img"
        img_dir = out_dir / "images" / out_name
        img_dir.mkdir(parents=True, exist_ok=True)
        records = []
        for ent_file in sorted(ent_dir.glob("*.txt")):
            stem = ent_file.stem
            src_img = img_src_dir / f"{stem}.jpg"
            if not src_img.exists():
                continue
            ent = json.loads(ent_file.read_text(errors="replace"))
            sample_id = f"{out_name}_{stem}"
            img_path = img_dir / f"{sample_id}.jpg"
            if not img_path.exists():
                shutil.copyfile(src_img, img_path)
            records.append({
                "id": sample_id,
                "image": str(img_path.relative_to(out_dir)),
                "source": "sroie",
                "gold": {
                    "vendor": ent.get("company"),
                    # SROIE dates are Malaysian receipts (overwhelmingly D/M/Y:
                    # 503:3 unambiguous-DMY:unambiguous-MDY in the raw data),
                    # so ambiguous numeric dates ("03/04/2024") are resolved
                    # DMY here rather than left as an un-normalizable raw
                    # string. Without this, gold falls back to raw text for
                    # ~35% of SROIE dates, which can never match a model
                    # trained (via distillation) to emit the ISO format the
                    # prompt actually asks for -- a real, structural date-EM
                    # ceiling that has nothing to do with reading accuracy.
                    "date": normalize_date(ent.get("date"), prefer_dmy=True),
                    "currency": None,
                    "total": normalize_number(ent.get("total")),
                    "tax": None,
                    "line_items": [],
                },
            })
        write_split(records, out_dir / f"{out_name}.jsonl")
        print(f"{out_name}: {len(records)} samples -> {out_dir / (out_name + '.jsonl')}")
