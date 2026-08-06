from ledgerlens.data.cord import _first_valid_number, _gt_to_extraction, _join_text


def test_join_text_single_string():
    assert _join_text("Coffee") == "Coffee"


def test_join_text_list_joins_all_not_just_first():
    # real CORD pattern: junk prefix line + actual name
    assert _join_text(["-", "Gyro Platter Large"]) == "- Gyro Platter Large"
    assert _join_text(["1", "MIE UP BSR C"]) == "1 MIE UP BSR C"


def test_join_text_none():
    assert _join_text(None) is None


def test_first_valid_number_plain():
    assert _first_valid_number("36,000") == 36000.0


def test_first_valid_number_skips_junk_prefix():
    # real CORD pattern: cnt: ["TRIPPLE CHEESE", "1"]
    assert _first_valid_number(["TRIPPLE CHEESE", "1"]) == 1.0


def test_first_valid_number_takes_first_of_multiple_readings():
    assert _first_valid_number(["55,834", "55,800"]) == 55834.0


def test_first_valid_number_none():
    assert _first_valid_number(None) is None
    assert _first_valid_number([]) is None


def test_gt_to_extraction_end_to_end_with_list_valued_fields():
    gt = {
        "menu": [
            {"nm": ["TAKEOUT(1):", "Viet Latte"], "cnt": "1", "price": "19.000"},
            {"nm": "Coffee", "cnt": ["2 pcs", "2"], "price": "7.00"},
        ],
        "total": {"total_price": ["55,834", "55,800"]},
        "sub_total": {"tax_price": "5,000"},
    }
    ex = _gt_to_extraction(gt)
    assert ex["line_items"][0]["description"] == "TAKEOUT(1): Viet Latte"
    assert ex["line_items"][1]["qty"] == 2.0  # skipped "2 pcs" junk, found "2"
    assert ex["total"] == 55834.0
    assert ex["tax"] == 5000.0
