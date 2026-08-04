"""Tests for the blend-policy logic in scripts/train_sft.py.

Pure Python, no torch/GPU needed -- but this logic decides exactly what data
the student model trains on, so it's worth locking down before any GPU run.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from train_sft import build_messages  # noqa: E402

GOLD = {"vendor": None, "date": "2024-01-01", "currency": None, "total": 10.0, "tax": None, "line_items": []}
TEACHER_VALID = {"vendor": "ACME", "date": "2024-01-01", "currency": "USD", "total": 10.0, "tax": 0.5, "line_items": []}
TEACHER_WRONG = {"vendor": "WRONG", "date": None, "currency": "EUR", "total": 999.0, "tax": None, "line_items": []}


def _rec(teacher_json, agrees):
    return {
        "image": "images/x.png", "prompt": "extract this",
        "gold": GOLD, "teacher_json": teacher_json, "teacher_agrees_total": agrees,
    }


def _target(msg):
    """Pull the assistant's JSON target back out of a build_messages() result."""
    text = msg["messages"][1]["content"][0]["text"]
    return json.loads(text)


def test_gold_policy_ignores_teacher_even_when_valid_and_agreeing():
    rec = _rec(TEACHER_VALID, agrees=True)
    assert _target(build_messages(rec, "gold")) == GOLD


def test_teacher_policy_uses_teacher_when_valid():
    rec = _rec(TEACHER_VALID, agrees=False)  # agreement irrelevant to "teacher" policy
    assert _target(build_messages(rec, "teacher")) == TEACHER_VALID


def test_teacher_policy_falls_back_to_gold_when_teacher_invalid():
    rec = _rec(None, agrees=False)  # teacher produced unparseable JSON
    assert _target(build_messages(rec, "teacher")) == GOLD


def test_teacher_agree_uses_teacher_only_when_agreeing():
    rec = _rec(TEACHER_VALID, agrees=True)
    assert _target(build_messages(rec, "teacher-agree")) == TEACHER_VALID


def test_teacher_agree_falls_back_to_gold_when_disagreeing():
    # teacher produced valid JSON but got the total wrong -- must not train on it
    rec = _rec(TEACHER_WRONG, agrees=False)
    assert _target(build_messages(rec, "teacher-agree")) == GOLD


def test_teacher_agree_falls_back_to_gold_when_teacher_invalid():
    rec = _rec(None, agrees=False)
    assert _target(build_messages(rec, "teacher-agree")) == GOLD


def test_unknown_policy_raises():
    rec = _rec(TEACHER_VALID, agrees=True)
    try:
        build_messages(rec, "bogus-policy")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_message_structure_has_image_and_prompt():
    rec = _rec(TEACHER_VALID, agrees=True)
    msg = build_messages(rec, "teacher-agree")
    assert msg["images"] == ["images/x.png"]
    assert msg["messages"][0]["role"] == "user"
    assert msg["messages"][0]["content"][0]["type"] == "image"
    assert msg["messages"][0]["content"][1]["text"] == "extract this"
    assert msg["messages"][1]["role"] == "assistant"
