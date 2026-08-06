from ledgerlens.normalize import normalize_currency, normalize_date, normalize_number, normalize_text


def test_number_us_style():
    assert normalize_number("1,234.56") == 1234.56


def test_number_eu_style():
    assert normalize_number("1.234,56") == 1234.56


def test_number_space_style():
    assert normalize_number("1 234,56") == 1234.56


def test_number_currency_prefix():
    assert normalize_number("$ 12.00") == 12.0
    assert normalize_number("Rp 15.000") == 15000.0  # IDR zero-decimal grouping


def test_number_plain_and_negative():
    assert normalize_number("1234") == 1234.0
    assert normalize_number("(4.50)") == -4.5
    assert normalize_number(7) == 7.0
    assert normalize_number(None) is None
    assert normalize_number("N/A") is None


def test_date_iso_and_textual():
    assert normalize_date("2024-03-05") == "2024-03-05"
    assert normalize_date("5 Mar 2024") == "2024-03-05"
    assert normalize_date("Mar 5, 2024") == "2024-03-05"


def test_date_unambiguous_numeric():
    # day > 12 forces d/m/y
    assert normalize_date("25/12/2018") == "2018-12-25"
    # month-first only valid interpretation
    assert normalize_date("12/25/2018") == "2018-12-25"


def test_date_ambiguous_not_guessed():
    # 03/04/2024 could be Mar 4 or Apr 3 — must NOT canonicalize
    assert normalize_date("03/04/2024") == "03/04/2024"


def test_date_prefer_dmy_resolves_ambiguous_case():
    assert normalize_date("03/04/2024", prefer_dmy=True) == "2024-04-03"


def test_date_prefer_dmy_does_not_affect_unambiguous_cases():
    assert normalize_date("25/12/2018", prefer_dmy=True) == "2018-12-25"
    assert normalize_date("12/25/2018", prefer_dmy=True) == "2018-12-25"


def test_date_month_name_dash_and_slash_variants():
    assert normalize_date("24-MAR-2018") == "2018-03-24"
    assert normalize_date("02/JAN/2017") == "2017-01-02"
    assert normalize_date("28 MAR 18") == "2018-03-28"


def test_currency():
    assert normalize_currency("$") == "USD"
    assert normalize_currency("Rp") == "IDR"
    assert normalize_currency("usd") == "USD"
    assert normalize_currency("EUR") == "EUR"


def test_text():
    assert normalize_text("  ACME   Corp. ") == "acme corp"
    assert normalize_text(None) is None
