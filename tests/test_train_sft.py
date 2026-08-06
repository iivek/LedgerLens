"""Tests for the field-level blend logic in scripts/train_sft.py.

Pure Python, no torch/GPU needed -- but this logic decides exactly what data
the student model trains on, so it's worth locking down before any GPU run.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from train_sft import build_messages, build_target  # noqa: E402

# CORD-style gold: vendor/date/currency are unannotated (always null in real
# CORD data), total/tax/line_items are annotated.
CORD_GOLD = {"vendor": None, "date": None, "currency": None, "total": 10.0, "tax": 0.5, "line_items": []}
TEACHER_VALID = {"vendor": "ACME", "date": "2024-01-01", "currency": "USD", "total": 10.0, "tax": 0.5, "line_items": []}
TEACHER_DISAGREES = {"vendor": "ACME", "date": "2024-01-01", "currency": "USD", "total": 999.0, "tax": 99.0, "line_items": []}

# SROIE-style gold: vendor/date/total annotated, currency/tax/line_items not.
SROIE_GOLD = {"vendor": "ACME Mart", "date": "2024-01-01", "currency": None, "total": 10.0, "tax": None, "line_items": []}


def _rec(source, gold, teacher_json):
    return {
        "image": "images/x.png", "prompt": "extract this", "source": source,
        "gold": gold, "teacher_json": teacher_json,
    }


def test_unannotated_fields_always_use_teacher_regardless_of_policy():
    # The core bug fix: CORD gold has vendor=None for every sample (CORD just
    # doesn't annotate it), but the receipt image visibly has a vendor. The
    # target must use the teacher's read there, not confidently supervise null
    # -- for ALL three policies, since there is no ground truth to be "gold
    # policy" about on this field.
    rec = _rec("cord", CORD_GOLD, TEACHER_VALID)
    for policy in ("gold", "teacher", "teacher-agree"):
        target = build_target(rec, policy)
        assert target["vendor"] == "ACME", f"policy={policy} failed"
        assert target["date"] == "2024-01-01", f"policy={policy} failed"
        assert target["currency"] == "USD", f"policy={policy} failed"


def test_unannotated_field_falls_back_to_none_when_teacher_invalid():
    rec = _rec("cord", CORD_GOLD, None)  # teacher produced invalid JSON
    target = build_target(rec, "teacher-agree")
    assert target["vendor"] is None  # no ground truth AND no valid teacher guess -- honestly null


def test_annotated_field_gold_policy_ignores_teacher():
    rec = _rec("cord", CORD_GOLD, TEACHER_DISAGREES)
    target = build_target(rec, "gold")
    assert target["total"] == 10.0  # gold's annotated value, not teacher's 999.0
    assert target["tax"] == 0.5


def test_annotated_field_teacher_policy_prefers_teacher_even_if_wrong():
    rec = _rec("cord", CORD_GOLD, TEACHER_DISAGREES)
    target = build_target(rec, "teacher")
    assert target["total"] == 999.0


def test_annotated_field_teacher_agree_uses_teacher_only_when_that_field_agrees():
    rec = _rec("cord", CORD_GOLD, TEACHER_VALID)  # total=10.0 matches gold
    target = build_target(rec, "teacher-agree")
    assert target["total"] == 10.0  # agrees -- uses teacher's (== gold's) value


def test_annotated_field_teacher_agree_falls_back_per_field_not_per_record():
    # Teacher agrees on total but disagrees on tax -- per-field check must
    # catch this independently, unlike the old whole-record total-only proxy.
    teacher = {**TEACHER_VALID, "tax": 55.0}  # total still matches gold (10.0), tax doesn't (0.5)
    rec = _rec("cord", CORD_GOLD, teacher)
    target = build_target(rec, "teacher-agree")
    assert target["total"] == 10.0   # total agreed -> teacher's value used
    assert target["tax"] == 0.5      # tax disagreed -> gold's value used instead


def test_line_items_agreement_uses_f1_not_just_total():
    gold = {**CORD_GOLD, "line_items": [{"description": "Coffee", "qty": 1, "unit_price": 3.5, "total": 3.5}]}
    teacher_bad_items = {**TEACHER_VALID, "total": 10.0,
                         "line_items": [{"description": "Totally Different Item", "qty": 9, "unit_price": 1, "total": 9}]}
    rec = _rec("cord", gold, teacher_bad_items)
    target = build_target(rec, "teacher-agree")
    assert target["line_items"] == gold["line_items"]  # teacher's items don't match -> gold used


def test_sroie_annotated_fields_are_the_complement_of_cord():
    rec = _rec("sroie", SROIE_GOLD, TEACHER_VALID)
    target = build_target(rec, "gold")
    assert target["vendor"] == "ACME Mart"  # SROIE annotates vendor -> gold used
    assert target["date"] == "2024-01-01"   # SROIE annotates date -> gold used
    assert target["currency"] == "USD"      # SROIE does NOT annotate currency -> teacher used regardless of policy
    assert target["tax"] == 0.5             # SROIE does NOT annotate tax -> teacher used


def test_unknown_policy_raises():
    rec = _rec("cord", CORD_GOLD, TEACHER_VALID)
    try:
        build_target(rec, "bogus-policy")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_message_structure_has_image_and_prompt():
    rec = _rec("cord", CORD_GOLD, TEACHER_VALID)
    msg = build_messages(rec, "teacher-agree")
    assert msg["images"] == ["images/x.png"]
    assert msg["messages"][0]["role"] == "user"
    assert msg["messages"][0]["content"][0]["type"] == "image"
    assert msg["messages"][0]["content"][1]["text"] == "extract this"
    assert msg["messages"][1]["role"] == "assistant"
    # target actually round-trips through the message as JSON
    target = json.loads(msg["messages"][1]["content"][0]["text"])
    assert target["vendor"] == "ACME"
