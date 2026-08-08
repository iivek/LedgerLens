import random

from faker import Faker

from ledgerlens.data.synthetic.generate import (
    _EVAL_CURRENCIES, _EVAL_ITEM_NOUNS, _TRAIN_CURRENCIES, _TRAIN_ITEM_NOUNS, make_sample,
)


def test_train_eval_pools_are_disjoint():
    # The actual guarantee that makes synth_eval a real generalization test
    # rather than a reshuffled copy of synth_train (a real bug, previously
    # unfixed: eval used to be a random 10% subset of the same distribution).
    train_currency_codes = {c[0] for c in _TRAIN_CURRENCIES}
    eval_currency_codes = {c[0] for c in _EVAL_CURRENCIES}
    assert not (train_currency_codes & eval_currency_codes)
    assert not (set(_TRAIN_ITEM_NOUNS) & set(_EVAL_ITEM_NOUNS))


def test_make_sample_train_split_only_uses_train_pools():
    rng = random.Random(1)
    faker = Faker(["en_US"])
    eval_currencies = {c[0] for c in _EVAL_CURRENCIES}
    for _ in range(30):
        _, ctx, code, _ = make_sample(rng, faker, split="train")
        assert code not in eval_currencies
        for item in ctx["items"]:
            assert item["description"] not in _EVAL_ITEM_NOUNS


def test_make_sample_eval_split_only_uses_eval_pools():
    rng = random.Random(1)
    faker = Faker(["en_US"])
    train_currencies = {c[0] for c in _TRAIN_CURRENCIES}
    for _ in range(30):
        _, ctx, code, _ = make_sample(rng, faker, split="eval")
        assert code not in train_currencies
        for item in ctx["items"]:
            assert item["description"] not in _TRAIN_ITEM_NOUNS


def test_gold_matches_rendered_totals():
    rng = random.Random(42)
    faker = Faker(["en_US"])
    _, ctx, _, _ = make_sample(rng, faker, split="train")
    gold = ctx["__gold__"]
    computed_subtotal = sum(i["total"] for i in gold["line_items"])
    computed_total = computed_subtotal + (gold["tax"] or 0)
    assert abs(computed_total - gold["total"]) < 0.01
