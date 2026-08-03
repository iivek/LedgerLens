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
