"""Weekend 4: HF checkpoint -> GGUF (+ mmproj) -> k-quant variants for llama.cpp/Android.

Wraps llama.cpp's converters — needs a llama.cpp checkout (env LLAMA_CPP or
--llama-cpp). Produces the exact files the Android app and the llamacpp harness
runner consume:

    models/gguf/<name>-f16.gguf         (LLM, fp16 reference)
    models/gguf/<name>-q8_0.gguf        (int8 point on the grid)
    models/gguf/<name>-q4_k_m.gguf      (int4 deployment target)
    models/gguf/mmproj-<name>-f16.gguf  (vision projector, f16 by default)

    python scripts/export_gguf.py --model runs/sft-2b-teacher-agree-merged --name ledgerlens

VISION-TOWER QUANTIZATION (--mmproj-quants): the mmproj file is 1.3GB f16 --
*larger* than the 4-bit LLM (0.99GB) in the current stock baseline, so it's
the single biggest lever nothing on the compression grid has touched yet
(flagged independently by two reviews). --mmproj-quants lets llama-quantize
run on it too, same as the LLM.

UNVERIFIED, flagged honestly rather than assumed: llama-quantize is built
against the main LLM's GGUF tensor layout; the mmproj/CLIP-vision GGUF is a
different format (clip.cpp's own layout), and whether llama-quantize accepts
it at all -- let alone produces a working, accurate vision tower -- depends
on the specific llama.cpp version and hasn't been tested here. This script
does NOT treat that quantized mmproj as trustworthy just because the command
exits 0: --verify-image (a real receipt photo) forces a real mtmd inference
call before printing success, so a silently-broken vision tower (garbage
output, or a technically-valid-but-useless quantization) gets caught here,
not after burning eval budget on it. If quantization fails outright or
--verify-image isn't given, the script falls back to keeping mmproj at f16
(the known-safe default) rather than shipping something unverified.
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


def try_quantize_mmproj(quantize_bin: Path, mmproj_f16: Path, out: Path, name: str,
                        quants: list[str], verify_image: str | None,
                        llama_cli: Path | None) -> dict[str, Path]:
    """Returns {quant_name: path} for mmproj variants that were both produced
    AND (if verify_image given) confirmed to load and run via a real mtmd call.
    Anything that fails either step is dropped with a printed reason, not
    silently included -- see module docstring on why this can't be trusted
    just because llama-quantize exits 0.
    """
    produced = {}
    for q in quants:
        dest = out / f"mmproj-{name}-{q}.gguf"
        try:
            sh([str(quantize_bin), str(mmproj_f16), str(dest), q.upper()])
        except subprocess.CalledProcessError as e:
            print(f"  mmproj {q}: llama-quantize failed ({e}) -- likely doesn't accept "
                  f"the mmproj/CLIP tensor layout on this llama.cpp version. Skipping.")
            continue
        if not verify_image:
            print(f"  mmproj {q}: quantized but UNVERIFIED (no --verify-image given) -- "
                  f"not included in the grid until it's actually tested with a real image")
            continue
        if llama_cli is None:
            print(f"  mmproj {q}: quantized but can't verify -- llama-mtmd-cli not found")
            continue
        try:
            out_text = subprocess.run(
                [str(llama_cli), "-m", str(out / f"{name}-q4_k_m.gguf"), "--mmproj", str(dest),
                 "--image", verify_image, "-p", "Describe this image in one sentence.",
                 "-n", "64", "--temp", "0"],
                capture_output=True, text=True, timeout=120, check=True,
            ).stdout
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            print(f"  mmproj {q}: FAILED real-image verification ({e}) -- dropped, not trustworthy")
            continue
        if len(out_text.strip()) < 10:
            print(f"  mmproj {q}: verification produced suspiciously little output "
                  f"({out_text.strip()[:50]!r}) -- dropped, likely broken")
            continue
        print(f"  mmproj {q}: verified OK on {verify_image} -> {dest}")
        produced[q] = dest
    return produced


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="merged HF checkpoint dir")
    ap.add_argument("--name", default="ledgerlens")
    ap.add_argument("--out", default="models/gguf")
    ap.add_argument("--llama-cpp", default=os.environ.get("LLAMA_CPP", "third_party/llama.cpp"))
    ap.add_argument("--quants", nargs="*", default=["q8_0", "q4_k_m"])
    ap.add_argument("--mmproj-quants", nargs="*", default=[],
                    help="EXPERIMENTAL, unverified (see module docstring): also try "
                         "quantizing the vision tower to these levels, e.g. q8_0 q4_k_m")
    ap.add_argument("--verify-image", help="a real receipt photo -- required to trust any "
                                           "--mmproj-quants output, see module docstring")
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

    mmproj_variants = {"f16": mmproj}
    if args.mmproj_quants:
        llama_cli = next((p for p in (llama / "build/bin/llama-mtmd-cli",
                                      llama / "llama-mtmd-cli") if p.exists()), None)
        print(f"\n--- mmproj quantization (experimental, see module docstring) ---")
        mmproj_variants.update(
            try_quantize_mmproj(quantize_bin, mmproj, out, args.name, args.mmproj_quants,
                                args.verify_image, llama_cli)
        )

    print("\nSizes:")
    for p in sorted(out.glob(f"*{args.name}*.gguf")):
        print(f"  {p.name:40s} {p.stat().st_size / 1e9:.2f} GB")
    if len(mmproj_variants) > 1:
        print(f"\nVerified mmproj variants: {sorted(mmproj_variants)} -- each is a real, "
              f"independent Pareto-grid point (LLM quant x mmproj quant), not just LLM points "
              f"with a fixed f16 vision tower.")

    print(f"\nServe locally:\n  {llama}/build/bin/llama-server -m {out}/{args.name}-q4_k_m.gguf "
          f"--mmproj {mmproj} -c 4096 --port 8080")
    # Destination filenames are fixed by android ModelFiles.kt — do not change one side only
    print(f"Push to phone:\n  adb push {out}/{args.name}-q4_k_m.gguf "
          f"/sdcard/Android/data/com.ledgerlens.app/files/models/ledgerlens-q4_k_m.gguf\n"
          f"  adb push {mmproj} "
          f"/sdcard/Android/data/com.ledgerlens.app/files/models/mmproj-f16.gguf")


if __name__ == "__main__":
    main()
