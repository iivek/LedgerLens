import random

import cv2
import numpy as np

from ledgerlens.data.synthetic.augment import augment_photo


def _text_image() -> np.ndarray:
    img = np.full((300, 400, 3), 255, dtype=np.uint8)
    cv2.putText(img, "TOTAL 1234.56", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    return img


def _legibility(img: np.ndarray) -> float:
    """Proxy for readability: edge energy after Canny (blurred/washed-out text has near-zero)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return float(edges.mean())


def test_augmented_images_stay_legible_across_seeds():
    base = _text_image()
    baseline_legibility = _legibility(base)
    worst = min(_legibility(augment_photo(base.copy(), random.Random(seed))) for seed in range(60))
    # Even the worst-case draw over many seeds must retain a meaningful fraction
    # of edge energy — guards against the degradations stacking to illegibility
    # (this reproduces a real bug: unrestrained stacking washed out all text).
    assert worst > baseline_legibility * 0.15


def test_deterministic_given_same_rng_state():
    base = _text_image()
    out1 = augment_photo(base.copy(), random.Random(123))
    out2 = augment_photo(base.copy(), random.Random(123))
    assert np.array_equal(out1, out2)


def test_output_shape_larger_than_input():
    base = _text_image()
    out = augment_photo(base.copy(), random.Random(1))
    assert out.shape[0] > base.shape[0] and out.shape[1] > base.shape[1]
