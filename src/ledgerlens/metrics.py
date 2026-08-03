"""Scoring: field-level EM/F1, JSON validity, Hungarian-matched line-item F1.

Fixed before training per the spec. All numbers published anywhere must come
from this module via the harness CLI.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from .normalize import normalize_currency, normalize_date, normalize_number, normalize_text
from .schema import NUMERIC_FIELDS, SCALAR_FIELDS, Extraction, LineItem

NUMERIC_TOLERANCE = 0.005  # absolute; totals compared after separator normalization
LINE_ITEM_MATCH_THRESHOLD = 0.5  # min pairwise similarity for a Hungarian match to count


def _normalize_scalar(name: str, value) -> object:
    if name in NUMERIC_FIELDS:
        return normalize_number(value)
    if name == "date":
        return normalize_date(value)
    if name == "currency":
        return normalize_currency(value)
    return normalize_text(value)


def scalar_match(name: str, pred, gold) -> bool:
    p, g = _normalize_scalar(name, pred), _normalize_scalar(name, gold)
    if p is None and g is None:
        return True
    if p is None or g is None:
        return False
    if name in NUMERIC_FIELDS:
        return abs(p - g) <= NUMERIC_TOLERANCE
    return p == g


def _token_f1(pred: str | None, gold: str | None) -> float:
    """Token-overlap F1 for strings (SQuAD-style); exact semantics for non-strings."""
    p, g = normalize_text(pred), normalize_text(gold)
    if p is None and g is None:
        return 1.0
    if p is None or g is None:
        return 0.0
    pt, gt = p.split(), g.split()
    common = sum(min(pt.count(t), gt.count(t)) for t in set(pt))
    if common == 0:
        return 0.0
    precision, recall = common / len(pt), common / len(gt)
    return 2 * precision * recall / (precision + recall)


def scalar_f1(name: str, pred, gold) -> float:
    if name in NUMERIC_FIELDS or name in ("date", "currency"):
        return 1.0 if scalar_match(name, pred, gold) else 0.0
    return _token_f1(pred, gold)


def _line_item_similarity(p: LineItem, g: LineItem) -> float:
    """Mean of description token-F1 and numeric agreement on (qty, unit_price, total)."""
    parts = [_token_f1(p.description, g.description)]
    for attr in ("qty", "unit_price", "total"):
        pv, gv = normalize_number(getattr(p, attr)), normalize_number(getattr(g, attr))
        if pv is None and gv is None:
            continue  # both absent: uninformative, don't inflate
        parts.append(1.0 if (pv is not None and gv is not None and abs(pv - gv) <= NUMERIC_TOLERANCE) else 0.0)
    return sum(parts) / len(parts)


def line_item_prf(pred: list[LineItem], gold: list[LineItem]) -> tuple[float, float, float]:
    """Precision/recall/F1 over line items with Hungarian (optimal) matching."""
    if not pred and not gold:
        return 1.0, 1.0, 1.0
    if not pred or not gold:
        return 0.0, 0.0, 0.0
    sim = np.array([[_line_item_similarity(p, g) for g in gold] for p in pred])
    rows, cols = linear_sum_assignment(-sim)
    matched = sum(1 for r, c in zip(rows, cols) if sim[r, c] >= LINE_ITEM_MATCH_THRESHOLD)
    precision, recall = matched / len(pred), matched / len(gold)
    f1 = 0.0 if matched == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


@dataclass
class SampleScore:
    sample_id: str
    json_valid: bool
    field_em: dict[str, float] = field(default_factory=dict)
    field_f1: dict[str, float] = field(default_factory=dict)
    line_item_p: float = 0.0
    line_item_r: float = 0.0
    line_item_f1: float = 0.0
    latency_s: float | None = None


def score_sample(sample_id: str, pred: Extraction | None, gold: Extraction,
                 latency_s: float | None = None) -> SampleScore:
    """Score one prediction. pred=None (invalid JSON) scores 0 everywhere."""
    if pred is None:
        return SampleScore(
            sample_id=sample_id, json_valid=False,
            field_em={f: 0.0 for f in SCALAR_FIELDS},
            field_f1={f: 0.0 for f in SCALAR_FIELDS},
            latency_s=latency_s,
        )
    em = {f: float(scalar_match(f, getattr(pred, f), getattr(gold, f))) for f in SCALAR_FIELDS}
    f1 = {f: scalar_f1(f, getattr(pred, f), getattr(gold, f)) for f in SCALAR_FIELDS}
    p, r, li_f1 = line_item_prf(pred.line_items, gold.line_items)
    return SampleScore(sample_id=sample_id, json_valid=True, field_em=em, field_f1=f1,
                       line_item_p=p, line_item_r=r, line_item_f1=li_f1, latency_s=latency_s)


def aggregate(scores: list[SampleScore]) -> dict[str, float]:
    """Corpus-level means; every published table row is one call to this."""
    if not scores:
        return {}
    n = len(scores)
    out: dict[str, float] = {
        "n_samples": float(n),
        "json_validity_rate": sum(s.json_valid for s in scores) / n,
    }
    for f in SCALAR_FIELDS:
        out[f"{f}_em"] = sum(s.field_em.get(f, 0.0) for s in scores) / n
        out[f"{f}_f1"] = sum(s.field_f1.get(f, 0.0) for s in scores) / n
    out["fields_em_mean"] = sum(out[f"{f}_em"] for f in SCALAR_FIELDS) / len(SCALAR_FIELDS)
    out["fields_f1_mean"] = sum(out[f"{f}_f1"] for f in SCALAR_FIELDS) / len(SCALAR_FIELDS)
    out["line_item_precision"] = sum(s.line_item_p for s in scores) / n
    out["line_item_recall"] = sum(s.line_item_r for s in scores) / n
    out["line_item_f1"] = sum(s.line_item_f1 for s in scores) / n
    latencies = [s.latency_s for s in scores if s.latency_s is not None]
    if latencies:
        out["latency_mean_s"] = float(np.mean(latencies))
        out["latency_p50_s"] = float(np.percentile(latencies, 50))
        out["latency_p95_s"] = float(np.percentile(latencies, 95))
    return out
