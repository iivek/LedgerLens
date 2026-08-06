from ledgerlens.metrics import aggregate, line_item_prf, scalar_match, score_sample
from ledgerlens.schema import Extraction, LineItem, parse_model_output

GOLD = Extraction(
    vendor="ACME Mart", date="2024-03-05", currency="USD", total=1234.56, tax=112.23,
    line_items=[
        LineItem(description="Coffee", qty=2, unit_price=3.5, total=7.0),
        LineItem(description="Notebook", qty=1, unit_price=12.0, total=12.0),
    ],
)


def test_perfect_prediction_scores_one():
    score = score_sample("s1", GOLD, GOLD)
    assert score.json_valid
    assert all(v == 1.0 for v in score.field_em.values())
    assert score.line_item_f1 == 1.0


def test_numeric_tolerance_and_separator():
    assert scalar_match("total", "1.234,56", 1234.56)
    assert scalar_match("total", 1234.555, 1234.56)  # within 0.005
    assert not scalar_match("total", 1235.0, 1234.56)


def test_both_null_matches():
    assert scalar_match("tax", None, None)
    assert not scalar_match("tax", None, 5.0)


def test_invalid_json_scores_zero():
    pred = parse_model_output("I think the total is around $12")
    score = score_sample("s2", pred, GOLD)
    assert not score.json_valid
    assert all(v == 0.0 for v in score.field_em.values())


def test_parse_with_fences_and_prose():
    raw = 'Sure! Here it is:\n```json\n{"vendor": "X", "total": 5.0, "line_items": []}\n```'
    ex = parse_model_output(raw)
    assert ex is not None and ex.vendor == "X" and ex.total == 5.0


def test_line_items_order_invariant():
    pred = [GOLD.line_items[1], GOLD.line_items[0]]  # swapped order
    p, r, f1 = line_item_prf(pred, GOLD.line_items)
    assert (p, r, f1) == (1.0, 1.0, 1.0)


def test_line_items_partial():
    pred = [GOLD.line_items[0]]  # missed one
    p, r, f1 = line_item_prf(pred, GOLD.line_items)
    assert p == 1.0 and r == 0.5


def test_line_items_hallucinated():
    pred = GOLD.line_items + [LineItem(description="Ghost item", qty=9, unit_price=99, total=891)]
    p, r, _ = line_item_prf(pred, GOLD.line_items)
    assert r == 1.0 and p < 1.0


def test_aggregate_shape():
    scores = [score_sample("a", GOLD, GOLD), score_sample("b", None, GOLD)]
    agg = aggregate(scores)
    assert agg["json_validity_rate"] == 0.5
    assert agg["vendor_em"] == 0.5
    assert 0 < agg["fields_f1_mean"] < 1


def test_unannotated_fields_not_penalized():
    # SROIE annotates only vendor/date/total: a model that also (correctly!)
    # extracts line items and currency must not lose points for it.
    gold = Extraction(vendor="ACME", date="2024-01-01", total=10.0)  # sroie-style gold
    pred = Extraction(vendor="ACME", date="2024-01-01", total=10.0, currency="MYR", tax=0.6,
                      line_items=[LineItem(description="Thing", qty=1, unit_price=10, total=10)])
    score = score_sample("s", pred, gold, source="sroie")
    assert set(score.field_em) == {"vendor", "date", "total"}
    assert all(v == 1.0 for v in score.field_em.values())
    assert score.line_item_f1 is None
    agg = aggregate([score])
    assert agg["fields_em_mean"] == 1.0
    assert "line_item_f1" not in agg
    assert "currency_em" not in agg


def test_invalid_json_zeroes_only_annotated_fields():
    gold = Extraction(total=10.0)
    score = score_sample("s", None, gold, source="cord")
    assert set(score.field_em) == {"total", "tax"}
    assert score.line_item_f1 == 0.0


def test_line_item_gold_null_numeric_not_penalized():
    # CORD-style gold: unit_price frequently unannotated. A model that
    # correctly extracts it anyway must not be scored as wrong for it.
    gold_item = LineItem(description="Coffee", qty=None, unit_price=None, total=7.0)
    pred_item = LineItem(description="Coffee", qty=2, unit_price=3.5, total=7.0)
    p, r, f1 = line_item_prf([pred_item], [gold_item])
    assert (p, r, f1) == (1.0, 1.0, 1.0)


def test_line_item_gold_present_pred_missing_still_penalized():
    # The fix must not become "gold-null skips everything" -- when gold DOES
    # have a value and the model fails to extract it, that's still wrong.
    gold_item = LineItem(description="Coffee", qty=2, unit_price=3.5, total=7.0)
    pred_item = LineItem(description="Coffee", qty=None, unit_price=None, total=7.0)
    p, r, f1 = line_item_prf([pred_item], [gold_item])
    # total matches, qty/unit_price both wrong -> similarity < 1 but still a match
    assert p == 1.0 and r == 1.0 and f1 == 1.0  # matched (similarity >= threshold)


def test_line_item_gold_present_pred_wrong_value_penalized():
    gold_item = LineItem(description="Coffee", qty=2, unit_price=3.5, total=7.0)
    pred_item = LineItem(description="Coffee", qty=99, unit_price=3.5, total=7.0)
    from ledgerlens.metrics import _line_item_similarity
    sim = _line_item_similarity(pred_item, gold_item)
    assert sim < 1.0  # qty wrong should still cost something
