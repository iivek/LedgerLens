"""Runner protocol: image + prompt -> raw text. One runner per deployment target.

Runners deliberately return *raw text*, not parsed Extractions — JSON validity is
a metric, so parsing happens in the harness where failures are counted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class ModelRunner(Protocol):
    name: str

    def generate(self, image_path: Path, prompt: str) -> str: ...


def get_runner(kind: str, **kwargs) -> ModelRunner:
    """Lazy imports so the local (CPU-only) env never imports torch."""
    if kind == "mock":
        from .mock import MockRunner
        return MockRunner(**kwargs)
    if kind == "hf":
        from .hf_vlm import HFVLMRunner
        return HFVLMRunner(**kwargs)
    if kind == "llamacpp":
        from .llamacpp import LlamaCppRunner
        return LlamaCppRunner(**kwargs)
    raise ValueError(f"unknown runner kind: {kind!r} (expected mock|hf|llamacpp)")
