"""Normalization applied to both predictions and ground truth before comparison.

The spec's numeric tolerance rule lives here: totals like "1.234,56", "1,234.56"
and "1234.56" must compare equal. Dates are canonicalized to ISO-8601 where the
format is unambiguous; ambiguous day/month orderings fall back to the raw string
so we never silently guess.
"""

from __future__ import annotations

import re
from datetime import datetime

_CURRENCY_ALIASES = {
    "$": "USD", "US$": "USD", "USD": "USD",
    "€": "EUR", "EUR": "EUR",
    "£": "GBP", "GBP": "GBP",
    "¥": "JPY", "JPY": "JPY",
    "₩": "KRW", "KRW": "KRW", "WON": "KRW",
    "₹": "INR", "INR": "INR", "RS": "INR", "RS.": "INR",
    "RM": "MYR", "MYR": "MYR",
    "S$": "SGD", "SGD": "SGD",
    "RP": "IDR", "IDR": "IDR",
    "₫": "VND", "VND": "VND",
    "₺": "TRY", "TRY": "TRY",
    "R$": "BRL", "BRL": "BRL",
    "₦": "NGN", "NGN": "NGN",
    "KSH": "KES", "KES": "KES",
    "AED": "AED", "PHP": "PHP", "THB": "THB", "CNY": "CNY", "RMB": "CNY",
}

_DATE_FORMATS_UNAMBIGUOUS = (
    "%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d",
    "%d %B %Y", "%d %b %Y", "%B %d, %Y", "%b %d, %Y", "%B %d %Y", "%b %d %Y",
    "%Y%m%d",
    # Month-name variants are unambiguous regardless of D/M ordering conventions
    # (a named month removes the question entirely) -- seen in real SROIE OCR
    # output ("24-MAR-2018", "02/JAN/2017", "28 MAR 18").
    "%d-%b-%Y", "%d/%b/%Y", "%d-%B-%Y", "%d/%B/%Y",
    "%d %b %y", "%d-%b-%y", "%d/%b/%y",
)
# Numeric little/middle-endian formats: only safe when one of day/month > 12
_DATE_FORMATS_DMY = ("%d-%m-%Y", "%d/%m/%Y", "%d.%m.%Y", "%d-%m-%y", "%d/%m/%y", "%d.%m.%y")
_DATE_FORMATS_MDY = ("%m-%d-%Y", "%m/%d/%Y", "%m.%d.%Y", "%m-%d-%y", "%m/%d/%y", "%m.%d.%y")


def normalize_text(value: str | None) -> str | None:
    """Casefold and collapse whitespace/punctuation noise for string comparison."""
    if value is None:
        return None
    v = re.sub(r"\s+", " ", value).strip().casefold()
    v = v.strip(" .,:;*-_")
    return v or None


def normalize_currency(value: str | None) -> str | None:
    if value is None:
        return None
    v = value.strip().upper()
    return _CURRENCY_ALIASES.get(v, v if re.fullmatch(r"[A-Z]{3}", v) else normalize_text(value))


# A single number token: a digit run with internal ./,  allowed, optionally
# continuing across a space/nbsp IFF what follows is a legitimate 3-digit
# thousands group (e.g. "1 234,56") -- so a stray word between two digit
# groups ("2 items 34.50") does not get silently spliced into one number.
_NUMBER_TOKEN_RE = re.compile(r"\(?\d[\d.,]*(?:[  ]\d{3}(?:[.,]\d+)?)*\)?")


def normalize_number(value: str | float | int | None) -> float | None:
    """Parse a monetary amount from messy strings; the spec's separator-tolerance rule.

    Handles "1,234.56", "1.234,56", "1 234,56", "$ 12.00", "12,00", "(4.50)" (negative).
    Returns None if no number can be extracted, OR if the string contains more
    than one disjoint number-like token (e.g. "2 items 34.50" from a garbled
    model output) -- concatenating unrelated digit groups into one confident
    wrong number is worse than declining to guess.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    raw = value.strip()
    negative = bool(re.match(r"^\(.*\)$", raw)) or raw.lstrip().startswith("-")
    tokens = _NUMBER_TOKEN_RE.findall(raw)
    if len(tokens) != 1:
        return None
    s = re.sub(r"[^\d.,]", "", tokens[0])
    if not re.search(r"\d", s):
        return None
    # Decide decimal separator: the rightmost of . or , iff followed by 1-2 digits at end
    last_dot, last_comma = s.rfind("."), s.rfind(",")
    sep_pos = max(last_dot, last_comma)
    if sep_pos != -1 and 1 <= len(s) - sep_pos - 1 <= 2:
        int_part = re.sub(r"[.,]", "", s[:sep_pos])
        dec_part = s[sep_pos + 1 :]
        num = f"{int_part}.{dec_part}"
    else:
        num = re.sub(r"[.,]", "", s)
    try:
        result = float(num)
    except ValueError:
        return None
    return -result if negative else result


def normalize_date(value: str | None, prefer_dmy: bool = False) -> str | None:
    """Canonicalize to YYYY-MM-DD when unambiguous; otherwise return cleaned raw string.

    By default, a numeric date where both D/M and M/D readings are valid
    (e.g. "03/04/2024") is left as raw text rather than guessed — the general
    case spans mixed-locale data (synthetic invoices, personal holdout) where
    guessing wrong is worse than declining to match.

    prefer_dmy=True resolves that same ambiguous case to the D/M reading
    instead of falling back to raw text. Only use this where the data source
    justifies a locale prior — e.g. SROIE (Malaysian receipts) is 503:3
    unambiguous-DMY:unambiguous-MDY in its raw dates, so DMY is the right
    default for the ~35% of its dates that are numeric-ambiguous. Using this
    generally would reintroduce the guessing risk this function exists to
    avoid; it's deliberately not the default.
    """
    if value is None:
        return None
    v = value.strip()
    for fmt in _DATE_FORMATS_UNAMBIGUOUS:
        try:
            return datetime.strptime(v, fmt).strftime("%Y-%m-%d")
        except ValueError:
            pass
    # Numeric d/m vs m/d: try both; accept only if exactly one interpretation is valid
    dmy = _try_formats(v, _DATE_FORMATS_DMY)
    mdy = _try_formats(v, _DATE_FORMATS_MDY)
    if dmy and mdy and dmy != mdy:
        return dmy if prefer_dmy else normalize_text(v)  # ambiguous — don't guess unless told to
    if dmy or mdy:
        return dmy or mdy
    return normalize_text(v)


def _try_formats(v: str, formats: tuple[str, ...]) -> str | None:
    for fmt in formats:
        try:
            return datetime.strptime(v, fmt).strftime("%Y-%m-%d")
        except ValueError:
            pass
    return None
