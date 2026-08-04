"""Personal holdout: ~50 real receipts photographed by the project author (spec §4) —
the honest "does it actually work" set, never used for training.

Unlike CORD/SROIE/synthetic there is no source of truth to convert; the human
has to supply it. Workflow:

    1. Drop photos in data/personal_holdout/raw/ (any image extension).
    2. Run `ledgerlens convert personal` once to scaffold a blank label JSON per
       photo in data/personal_holdout/labels/ (matching Extraction's schema).
    3. Hand-fill each label JSON with what the receipt actually says.
    4. Run `ledgerlens convert personal` again to build the unified split from
       whichever labels are now filled in (partially-labeled sets are fine —
       scaffolds with all-null fields are skipped with a warning, not treated
       as "this receipt has no vendor").

Redact or exclude any privacy-sensitive photos before publishing anything
derived from this split (spec §4).
"""

from __future__ import annotations

import json
from pathlib import Path

from ..schema import Extraction
from .unified import write_split

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".heic", ".webp")
BLANK_LABEL = {
    "vendor": None, "date": None, "currency": None,
    "total": None, "tax": None, "line_items": [],
}


def _is_blank(label: dict) -> bool:
    return label == BLANK_LABEL


def _check_no_unknown_keys(label: dict, label_path: Path) -> None:
    # Extraction ignores unknown fields by default (deliberately, so it stays
    # tolerant of stray fields in model output) -- but that leniency is wrong
    # for hand-typed labels, where a typo'd key should fail loudly rather than
    # silently vanish into an all-null field.
    unknown = set(label) - set(BLANK_LABEL)
    if unknown:
        raise ValueError(
            f"{label_path}: unknown field(s) {sorted(unknown)} -- typo? "
            f"expected only {sorted(BLANK_LABEL)}"
        )


def convert_personal(out_dir: Path, raw_dir: Path | None = None, labels_dir: Path | None = None) -> None:
    raw_dir = raw_dir or out_dir / "personal_holdout" / "raw"
    labels_dir = labels_dir or out_dir / "personal_holdout" / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    if not raw_dir.exists():
        raise SystemExit(
            f"No photos found at {raw_dir}.\n"
            "Photograph ~50 real receipts/invoices (spec §4), drop them there, and rerun."
        )
    images = sorted(p for p in raw_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        raise SystemExit(f"{raw_dir} exists but has no image files ({IMAGE_EXTS}).")

    scaffolded = []
    for img in images:
        label_path = labels_dir / f"{img.stem}.json"
        if not label_path.exists():
            label_path.write_text(json.dumps(BLANK_LABEL, indent=2) + "\n")
            scaffolded.append(label_path)

    if scaffolded:
        print(f"Scaffolded {len(scaffolded)} blank label file(s) in {labels_dir}:")
        for p in scaffolded:
            print(f"  {p.name}")
        print("\nFill in what each receipt actually says, then rerun `ledgerlens convert personal`.")

    records = []
    skipped_blank = 0
    for img in images:
        label_path = labels_dir / f"{img.stem}.json"
        label = json.loads(label_path.read_text())
        if _is_blank(label):
            skipped_blank += 1
            continue
        _check_no_unknown_keys(label, label_path)
        gold = Extraction.model_validate(label).model_dump()
        records.append({
            "id": f"personal_{img.stem}",
            "image": str((out_dir / "images" / "personal" / img.name).relative_to(out_dir)),
            "source": "personal",
            "gold": gold,
        })
        dst = out_dir / "images" / "personal" / img.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            dst.write_bytes(img.read_bytes())

    if records:
        write_split(records, out_dir / "personal_holdout.jsonl")
        print(f"\npersonal_holdout: {len(records)} labeled samples -> {out_dir / 'personal_holdout.jsonl'}")
    if skipped_blank:
        print(f"({skipped_blank} label(s) still blank, skipped -- fill them in and rerun)")
