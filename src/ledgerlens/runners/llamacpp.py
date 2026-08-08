"""llama.cpp runner: talks to a running llama-server (OpenAI-compatible API).

This is the edge-deployment path: the same GGUF + mmproj that ships to the phone,
served locally, e.g.:

    llama-server -m models/ledgerlens-q4_k_m.gguf --mmproj models/mmproj-f16.gguf \
        -c 4096 --port 8080

Using the server (rather than in-process bindings) keeps the harness measuring the
same runtime the phone uses, and needs no Python GPU deps.
"""

from __future__ import annotations

import base64
import hashlib
import json
import urllib.error
import urllib.request
from pathlib import Path

from ..schema import MAX_NEW_TOKENS


def _sha256(path: str) -> str | None:
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]  # short form; full CSV rows shouldn't be enormous


class LlamaCppRunner:
    def __init__(self, base_url: str = "http://127.0.0.1:8080", name: str = "llamacpp",
                 max_new_tokens: int = MAX_NEW_TOKENS, timeout_s: float = 600.0,
                 model_path: str | None = None, mmproj_path: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.name = name
        self.max_new_tokens = max_new_tokens
        self.timeout_s = timeout_s
        # Provenance-only: llama-server already has the model loaded, these
        # aren't used for requests, just to record what was actually running --
        # `tag` used to be the only record of which model/quant a row measured,
        # a human-typed string with no way to verify it.
        self.model_path = model_path
        self.mmproj_path = mmproj_path

    def provenance(self) -> dict:
        prov = {
            "runtime": "llama.cpp/llama-server",
            "max_new_tokens": self.max_new_tokens,
            "repeat_penalty": 1.0,
        }
        if self.model_path:
            prov["model_path"] = self.model_path
            prov["model_sha256_16"] = _sha256(self.model_path)
            prov["model_size_bytes"] = Path(self.model_path).stat().st_size if Path(self.model_path).is_file() else None
        if self.mmproj_path:
            prov["mmproj_path"] = self.mmproj_path
            prov["mmproj_sha256_16"] = _sha256(self.mmproj_path)
        try:
            req = urllib.request.Request(f"{self.base_url}/props")
            with urllib.request.urlopen(req, timeout=10) as resp:
                props = json.load(resp)
            # /props shape varies by llama.cpp build; record what's there
            # rather than assume specific keys exist.
            prov["server_props"] = json.dumps(props, ensure_ascii=False)[:2000]
        except (urllib.error.URLError, TimeoutError, OSError):
            pass  # older builds may not expose /props; provenance stays partial, not absent
        return prov

    def generate(self, image_path: Path, prompt: str) -> str:
        b64 = base64.b64encode(image_path.read_bytes()).decode()
        suffix = image_path.suffix.lstrip(".").lower() or "jpeg"
        payload = {
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/{suffix};base64,{b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }],
            "max_tokens": self.max_new_tokens,
            "temperature": 0.0,
            # llama-server's repeat_penalty commonly defaults to a non-neutral
            # value (e.g. 1.1), which actively penalizes repeated tokens --
            # structured JSON output is inherently repetitive (braces, quotes,
            # field names), so an unpinned default sampler isn't actually the
            # "fixed prompt, greedy decoding" comparison this harness claims.
            # Pinned explicitly rather than trusted to the server's default.
            "repeat_penalty": 1.0,
        }
        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            body = json.load(resp)
        return body["choices"][0]["message"]["content"]
