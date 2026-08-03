"""Weekend 4: HF checkpoint -> GGUF (+ mmproj) -> k-quant variants for llama.cpp/Android.

Wraps llama.cpp's converters — needs a llama.cpp checkout (env LLAMA_CPP or
--llama-cpp). Produces the exact files the Android app and the llamacpp harness
runner consume:

    models/gguf/<name>-f16.gguf         (LLM, fp16 reference)
    models/gguf/<name>-q8_0.gguf        (int8 point on the grid)
    models/gguf/<name>-q4_k_m.gguf      (int4 deployment target)
    models/gguf/mmproj-<name>-f16.gguf  (vision projector, stays fp16)

    python scripts/export_gguf.py --model runs/sft-2b-teacher-agree-merged --name ledgerlens
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def sh(cmd: list[str]) -> None:
    print("+", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="merged HF checkpoint dir")
    ap.add_argument("--name", default="ledgerlens")
    ap.add_argument("--out", default="models/gguf")
    ap.add_argument("--llama-cpp", default=os.environ.get("LLAMA_CPP", "third_party/llama.cpp"))
    ap.add_argument("--quants", nargs="*", default=["q8_0", "q4_k_m"])
    args = ap.parse_args()

    llama = Path(args.llama_cpp)
    convert = llama / "convert_hf_to_gguf.py"
    quantize_bin = next((p for p in (llama / "build/bin/llama-quantize",
                                     llama / "llama-quantize") if p.exists()), None)
    if not convert.exists():
        sys.exit(f"llama.cpp checkout not found at {llama} (set --llama-cpp or LLAMA_CPP)")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    f16 = out / f"{args.name}-f16.gguf"
    mmproj = out / f"mmproj-{args.name}-f16.gguf"
    # Qwen2-VL: one pass for the LLM, one with --mmproj for the vision projector
    sh([sys.executable, str(convert), args.model, "--outfile", str(f16), "--outtype", "f16"])
    sh([sys.executable, str(convert), args.model, "--outfile", str(mmproj),
        "--outtype", "f16", "--mmproj"])

    if quantize_bin is None:
        sys.exit("llama-quantize binary not found — build llama.cpp first (cmake --build build)")
    for q in args.quants:
        sh([str(quantize_bin), str(f16), str(out / f"{args.name}-{q}.gguf"), q.upper()])

    print("\nSizes:")
    for p in sorted(out.glob(f"*{args.name}*.gguf")):
        print(f"  {p.name:40s} {p.stat().st_size / 1e9:.2f} GB")
    print(f"\nServe locally:\n  {llama}/build/bin/llama-server -m {out}/{args.name}-q4_k_m.gguf "
          f"--mmproj {mmproj} -c 4096 --port 8080")
    print(f"Push to phone:\n  adb push {out}/{args.name}-q4_k_m.gguf "
          f"/sdcard/Android/data/com.ledgerlens.app/files/models/ledgerlens-q4_k_m.gguf")


if __name__ == "__main__":
    main()
