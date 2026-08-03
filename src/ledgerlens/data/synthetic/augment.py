"""Phone-photo augmentations: perspective warp, blur, shadow, low-light, noise.

Turns clean renders into things that look like a receipt photographed on a table.
Deterministic given the caller's random.Random instance.
"""

from __future__ import annotations

import random

import cv2
import numpy as np


# Independent per-effect probabilities each look reasonable in isolation, but
# stacking e.g. heavy motion blur + deep shadow + low light + strong JPEG
# artifacts compounds into text that is illegible even to a human — which would
# teach the model to hallucinate the gold JSON rather than read it. Each active
# effect below draws its own severity in [0, 1] and is scaled down once the
# cumulative severity budget is spent, so realistic single/double degradations
# stay full-strength while five-at-once cases get capped rather than removed.
SEVERITY_BUDGET = 1.6


def augment_photo(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """img: BGR uint8. Applies a random subset of realistic degradations."""
    img = _pad_on_background(img, rng)
    img = _perspective(img, rng)
    spent = 0.0
    for prob, fn in (
        (0.7, _shadow), (0.5, _lighting), (0.6, _blur), (0.5, _noise), (0.7, _jpeg),
    ):
        if rng.random() < prob:
            severity = rng.random()
            scale = min(1.0, max(0.15, SEVERITY_BUDGET - spent))
            img = fn(img, rng, severity * scale)
            spent += severity
    return img


def _pad_on_background(img: np.ndarray, rng: random.Random) -> np.ndarray:
    h, w = img.shape[:2]
    pad = int(min(h, w) * rng.uniform(0.06, 0.18))
    bg_color = np.array([rng.randint(40, 140)] * 3) + np.array(
        [rng.randint(-20, 20), rng.randint(-20, 20), rng.randint(-20, 20)]
    )
    bg = np.clip(bg_color, 0, 255).astype(np.uint8)
    canvas = np.full((h + 2 * pad, w + 2 * pad, 3), bg, dtype=np.uint8)
    # subtle background texture
    noise = np.random.default_rng(rng.randrange(2**32)).normal(0, 6, canvas.shape)
    canvas = np.clip(canvas.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    canvas[pad : pad + h, pad : pad + w] = img
    return canvas


def _perspective(img: np.ndarray, rng: random.Random) -> np.ndarray:
    h, w = img.shape[:2]
    d = 0.06
    jitter = lambda: rng.uniform(-d, d)  # noqa: E731
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([
        [w * jitter(), h * jitter()],
        [w * (1 + jitter()), h * jitter()],
        [w * (1 + jitter()), h * (1 + jitter())],
        [w * jitter(), h * (1 + jitter())],
    ])
    m = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, m, (w, h), borderMode=cv2.BORDER_REPLICATE)


def _shadow(img: np.ndarray, rng: random.Random, severity: float) -> np.ndarray:
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    pts = np.array([[rng.randint(0, w), rng.randint(0, h)] for _ in range(4)])
    opacity = 0.15 + severity * 0.4  # up to 0.55 at full severity
    cv2.fillConvexPoly(mask, cv2.convexHull(pts), opacity)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=min(h, w) * 0.08)
    return np.clip(img.astype(np.float32) * (1 - mask[..., None]), 0, 255).astype(np.uint8)


def _lighting(img: np.ndarray, rng: random.Random, severity: float) -> np.ndarray:
    # Interpolate toward neutral (alpha=1, beta=0) as severity drops.
    alpha = 1.0 + (rng.uniform(0.55, 1.1) - 1.0) * severity
    beta = rng.uniform(-35, 10) * severity
    warm = rng.uniform(-12, 12) * severity  # cheap white-balance shift
    out = img.astype(np.float32) * alpha + beta
    out[..., 2] += warm  # BGR: red channel
    out[..., 0] -= warm
    return np.clip(out, 0, 255).astype(np.uint8)


def _blur(img: np.ndarray, rng: random.Random, severity: float) -> np.ndarray:
    if rng.random() < 0.5:  # motion blur
        k = rng.choice([3, 5, 7, 9] if severity > 0.6 else [3, 5])
        kernel = np.zeros((k, k), dtype=np.float32)
        kernel[k // 2, :] = 1.0 / k
        angle = rng.uniform(0, 180)
        m = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), angle, 1)
        kernel = cv2.warpAffine(kernel, m, (k, k))
        kernel /= max(kernel.sum(), 1e-6)
        return cv2.filter2D(img, -1, kernel)
    return cv2.GaussianBlur(img, (0, 0), sigmaX=0.4 + severity * 1.2)


def _noise(img: np.ndarray, rng: random.Random, severity: float) -> np.ndarray:
    sigma = 2 + severity * 8
    noise = np.random.default_rng(rng.randrange(2**32)).normal(0, sigma, img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def _jpeg(img: np.ndarray, rng: random.Random, severity: float) -> np.ndarray:
    q = round(90 - severity * 35)  # 90 (mild) down to 55 (full severity)
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR) if ok else img
