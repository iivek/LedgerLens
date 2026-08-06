"""The extraction contract: one JSON schema shared by prompts, metrics, training and the app.

Every model (teacher, student, quantized variants, the Android runtime) is asked for
exactly this JSON. The eval harness scores against it. Changing this schema after
training starts invalidates comparisons — treat it as frozen once baselines are run.
"""

from __future__ import annotations

import json
import re

from pydantic import BaseModel, ValidationError


class LineItem(BaseModel):
    description: str | None = None
    qty: float | None = None
    unit_price: float | None = None
    total: float | None = None


class Extraction(BaseModel):
    vendor: str | None = None
    date: str | None = None  # ISO-8601 YYYY-MM-DD after normalization
    currency: str | None = None  # ISO-4217 code, e.g. "USD"
    total: float | None = None
    tax: float | None = None
    line_items: list[LineItem] = []


# Canonical generation/image-preprocessing config -- the single source every
# Python code path (hf_vlm.py, llamacpp.py, train_sft.py, prune.py) and the
# Android app should match. Previously these diverged silently: the harness
# capped generation at 512 tokens while Android used 1024 (part of why a
# 22-line-item CORD receipt truncated during teacher labeling), and
# min/max_pixels varied between eval (1280*28*28), training (1024*28*28),
# and pruning (uncapped/library default) -- three different effective image
# resolutions across stages that were never actually a controlled comparison.
# CHANGE NOTE: Weekend-1 stock baselines (results.csv tags without
# "-rescored") were measured under the OLD 512-token cap, before this was
# unified to 1024 to match Android. Not retroactively changed; flagged here
# so a future reader knows why an old row's config differs from a new one.
MAX_NEW_TOKENS = 1024
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28


# Scalar fields scored individually by the harness (line_items scored separately).
SCALAR_FIELDS = ("vendor", "date", "currency", "total", "tax")
NUMERIC_FIELDS = ("total", "tax")

PROMPT = """Extract the following fields from this receipt or invoice image and answer with a single JSON object, nothing else:
{"vendor": string|null, "date": "YYYY-MM-DD"|null, "currency": ISO-4217 string|null, "total": number|null, "tax": number|null, "line_items": [{"description": string, "qty": number, "unit_price": number, "total": number}]}
Use null for fields not present. Do not invent values."""

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def parse_model_output(text: str) -> Extraction | None:
    """Parse raw model text into an Extraction, or None if invalid.

    Tolerates markdown fences and leading/trailing prose (takes the first
    balanced top-level JSON object). None => counts against JSON-validity rate.
    """
    if not text:
        return None
    cleaned = _FENCE_RE.sub("", text.strip()).strip()
    candidate = _first_json_object(cleaned)
    if candidate is None:
        return None
    try:
        return Extraction.model_validate(json.loads(candidate))
    except (json.JSONDecodeError, ValidationError):
        return None


def _first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None
