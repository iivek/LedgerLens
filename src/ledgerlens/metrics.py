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

# Datasets only score the fields they actually annotate: an unannotated field is
# stored as null in gold, which must mean "unknown", never "wrong if predicted".
# Fixed alongside the metrics, before any baseline row is recorded.
_ALL_FIELDS = set(SCALAR_FIELDS) | {"line_items"}
ANNOTATED_FIELDS: dict[str, set[str]] = {
    "cord": {"total", "tax", "line_items"},
    "sroie": {"vendor", "date", "total"},
    "synthetic": _ALL_FIELDS,
    "personal": _ALL_FIELDS,
}


def annotated_fields(source: str) -> set[str]:
    return ANNOTATED_FIELDS.get(source, _ALL_FIELDS)


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
    """Mean of description token-F1 and numeric agreement on (qty, unit_price, total).

    Gold null on a numeric attribute means "unannotated" (CORD's line items
    frequently lack unit_price/qty in the source data), not "this line item
    has no such value" -- the same unannotated-vs-wrong distinction already
    applied to top-level fields via ANNOTATED_FIELDS. A gold-null attribute is
    skipped regardless of what the model predicted, so a correct extraction of
    a field the annotator simply didn't record isn't scored as wrong. A model
    that fails to extract an attribute gold DOES have is still scored 0.0.
    Verified against real predictions.jsonl: this was previously understating
    line-item F1 by ~9-10pp on CORD, where unit_price is null in ~73% of gold
    line items.
    """
    parts = [_token_f1(p.description, g.description)]
    for attr in ("qty", "unit_price", "total"):
        pv, gv = normalize_number(getattr(p, attr)), normalize_number(getattr(g, attr))
        if gv is None:
            continue  # unannotated in gold: uninformative, don't penalize either way
        parts.append(1.0 if (pv is not None and abs(pv - gv) <= NUMERIC_TOLERANCE) else 0.0)
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
    # Only annotated fields appear in these dicts; line_item_* is None when the
    # sample's source doesn't annotate line items.
    field_em: dict[str, float] = field(default_factory=dict)
    field_f1: dict[str, float] = field(default_factory=dict)
    line_item_p: float | None = None
    line_item_r: float | None = None
    line_item_f1: float | None = None
    latency_s: float | None = None


def score_sample(sample_id: str, pred: Extraction | None, gold: Extraction,
                 latency_s: float | None = None, source: str = "unknown") -> SampleScore:
    """Score one prediction over the fields its source annotates.

    pred=None (invalid JSON) scores 0 on every annotated field.
    """
    fields = annotated_fields(source)
    scalar = [f for f in SCALAR_FIELDS if f in fields]
    score_items = "line_items" in fields
    if pred is None:
        return SampleScore(
            sample_id=sample_id, json_valid=False,
            field_em={f: 0.0 for f in scalar},
            field_f1={f: 0.0 for f in scalar},
            line_item_p=0.0 if score_items else None,
            line_item_r=0.0 if score_items else None,
            line_item_f1=0.0 if score_items else None,
            latency_s=latency_s,
        )
    em = {f: float(scalar_match(f, getattr(pred, f), getattr(gold, f))) for f in scalar}
    f1 = {f: scalar_f1(f, getattr(pred, f), getattr(gold, f)) for f in scalar}
    p = r = li_f1 = None
    if score_items:
        p, r, li_f1 = line_item_prf(pred.line_items, gold.line_items)
    return SampleScore(sample_id=sample_id, json_valid=True, field_em=em, field_f1=f1,
                       line_item_p=p, line_item_r=r, line_item_f1=li_f1, latency_s=latency_s)


def aggregate(scores: list[SampleScore]) -> dict[str, float]:
    """Corpus-level means; every published table row is one call to this.

    Each field averages over the samples that annotate it, so mixed-source splits
    stay fair; `<field>_n` records the denominator.
    """
    if not scores:
        return {}
    n = len(scores)
    out: dict[str, float] = {
        "n_samples": float(n),
        "json_validity_rate": sum(s.json_valid for s in scores) / n,
    }
    field_means = []
    for f in SCALAR_FIELDS:
        with_f = [s for s in scores if f in s.field_em]
        if not with_f:
            continue
        out[f"{f}_em"] = sum(s.field_em[f] for s in with_f) / len(with_f)
        out[f"{f}_f1"] = sum(s.field_f1[f] for s in with_f) / len(with_f)
        out[f"{f}_n"] = float(len(with_f))
        field_means.append(f)
    if field_means:
        out["fields_em_mean"] = sum(out[f"{f}_em"] for f in field_means) / len(field_means)
        out["fields_f1_mean"] = sum(out[f"{f}_f1"] for f in field_means) / len(field_means)
    with_items = [s for s in scores if s.line_item_f1 is not None]
    if with_items:
        out["line_item_precision"] = sum(s.line_item_p for s in with_items) / len(with_items)
        out["line_item_recall"] = sum(s.line_item_r for s in with_items) / len(with_items)
        out["line_item_f1"] = sum(s.line_item_f1 for s in with_items) / len(with_items)
        out["line_item_n"] = float(len(with_items))
    latencies = [s.latency_s for s in scores if s.latency_s is not None]
    if latencies:
        out["latency_mean_s"] = float(np.mean(latencies))
        out["latency_p50_s"] = float(np.percentile(latencies, 50))
        out["latency_p95_s"] = float(np.percentile(latencies, 95))
    return out
