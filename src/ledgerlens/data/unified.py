"""Unified dataset format: one JSONL file per split + an images/ directory.

Every source (CORD, SROIE, synthetic, personal holdout) converts into this, so the
harness, teacher-labeling and SFT scripts never care where a sample came from.

Line format:
    {"id": str, "image": str (path relative to the jsonl), "source": str,
     "gold": {<Extraction fields>}}
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..schema import Extraction


@dataclass
class Sample:
    id: str
    image: Path  # absolute, resolved at load time
    source: str
    gold: Extraction


def load_split(jsonl_path: str | Path) -> list[Sample]:
    jsonl_path = Path(jsonl_path)
    base = jsonl_path.parent
    samples = []
    with open(jsonl_path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            samples.append(Sample(
                id=rec["id"],
                image=(base / rec["image"]).resolve(),
                source=rec.get("source", "unknown"),
                gold=Extraction.model_validate(rec["gold"]),
            ))
    return samples


def write_split(samples: list[dict], jsonl_path: str | Path) -> None:
    """samples: dicts with id/image (relative str)/source/gold (dict)."""
    jsonl_path = Path(jsonl_path)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, "w") as f:
        for rec in samples:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
