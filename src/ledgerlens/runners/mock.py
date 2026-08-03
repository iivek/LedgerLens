"""Deterministic mock runner: echoes gold (optionally corrupted) for harness tests.

Lets the whole eval pipeline run end-to-end on any machine with no model, and
gives tests known-accuracy fixtures (accuracy=1.0 uncorrupted; degraded modes
exercise each metric path).
"""

from __future__ import annotations

import json
import random
from pathlib import Path


class MockRunner:
    name = "mock"

    def __init__(self, gold_by_image: dict[str, dict] | None = None,
                 corrupt_rate: float = 0.0, invalid_json_rate: float = 0.0, seed: int = 0):
        self.gold_by_image = gold_by_image or {}
        self.corrupt_rate = corrupt_rate
        self.invalid_json_rate = invalid_json_rate
        self.rng = random.Random(seed)

    def generate(self, image_path: Path, prompt: str) -> str:
        gold = self.gold_by_image.get(str(image_path), {})
        if self.rng.random() < self.invalid_json_rate:
            return "Sorry, I can't read this receipt {"
        out = json.loads(json.dumps(gold))  # deep copy
        if self.rng.random() < self.corrupt_rate:
            out["total"] = (out.get("total") or 0) + 1.0
            out["vendor"] = "WRONG VENDOR LLC"
        return json.dumps(out)
