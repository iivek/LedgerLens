"""Efficiency measurements for the Pareto grid: size, peak RAM, tok/s, e2e latency.

Accuracy comes from `ledgerlens eval`; this script produces the efficiency columns
for the same tag, appended to results/efficiency.csv. Two modes:

  gguf — benchmarks a GGUF via llama.cpp's llama-mtmd-cli under /usr/bin/time -l
         (macOS) or -v (Linux) for true peak-RSS, plus timing parsed from output.
         This is the mode used on laptop-CPU; on-phone numbers come from the app's
         own timer + adb dumpsys meminfo, entered with --manual.
  hf   — benchmarks an HF checkpoint on GPU (torch.cuda.max_memory_allocated).

    python scripts/bench_efficiency.py gguf --model models/gguf/ledgerlens-q4_k_m.gguf \
        --mmproj models/gguf/mmproj-ledgerlens-f16.gguf --image samples/receipt.jpg \
        --tag student2b-q4km --host laptop-cpu
    python scripts/bench_efficiency.py manual --tag student2b-q4km --host phone-pixel6a \
        --size-gb 1.1 --peak-ram-gb 2.4 --tok-s 11.2 --e2e-s 8.9
"""

from __future__ import annotations

import argparse
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from ledgerlens.harness import append_results_csv  # noqa: E402
from ledgerlens.schema import PROMPT  # noqa: E402

CSV = Path("results/efficiency.csv")


def _row(tag: str, host: str, **kw) -> None:
    row = {"tag": tag, "runner": kw.pop("runner", ""), "split": "", "host": host,
           "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
           **{k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in kw.items()}}
    append_results_csv(CSV, row)
    print("\n".join(f"{k}: {v}" for k, v in row.items()))


def parse_tok_s(stderr_text: str) -> float | None:
    """llama.cpp timing line: "eval time = ... ms / N tokens ( ... ms per token, X tokens per second)"."""
    m = re.findall(r"([\d.]+)\s+tokens per second", stderr_text)
    return float(m[-1]) if m else None


def parse_peak_ram_gb(stderr_text: str, is_darwin: bool) -> float | None:
    """Peak RSS from `/usr/bin/time -l` (macOS, bytes) or `-v` (Linux, kbytes)."""
    if is_darwin:
        m = re.search(r"(\d+)\s+maximum resident set size", stderr_text)
        return int(m.group(1)) / 1e9 if m else None
    m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", stderr_text)
    return int(m.group(1)) * 1024 / 1e9 if m else None


def bench_gguf(args: argparse.Namespace) -> None:
    size_gb = Path(args.model).stat().st_size / 1e9
    mmproj_gb = Path(args.mmproj).stat().st_size / 1e9
    cli = args.cli or "llama-mtmd-cli"
    base = [cli, "-m", args.model, "--mmproj", args.mmproj, "--image", args.image,
            "-p", PROMPT, "-n", str(args.max_tokens), "--temp", "0"]
    is_darwin = platform.system() == "Darwin"
    time_cmd = ["/usr/bin/time", "-l"] if is_darwin else ["/usr/bin/time", "-v"]

    runs = []
    peak_ram_gb = None
    for i in range(args.repeats):
        t0 = time.perf_counter()
        proc = subprocess.run(time_cmd + base, capture_output=True, text=True)
        e2e = time.perf_counter() - t0
        err = proc.stderr
        tok_s = parse_tok_s(err)
        ram = parse_peak_ram_gb(err, is_darwin)
        if ram is not None:
            # max across repeats, not last-run-wins: peak RAM is what matters
            # for "will this fit on the phone", and later runs (warm caches,
            # less page-in) can under-report relative to the true worst case.
            peak_ram_gb = max(peak_ram_gb or 0.0, ram)
        runs.append((e2e, tok_s))
        print(f"run {i + 1}/{args.repeats}: e2e={e2e:.2f}s tok/s={tok_s}")

    warm = runs[1:] or runs  # first run pays model load + mmap page-in
    tok_s_vals = [t for _, t in warm if t]
    # Some llama-mtmd-cli builds (this is an "experimental CLI for testing
    # multimodal capability" per its own startup warning) never print a
    # "tokens per second" line at all, even with --perf -- confirmed directly,
    # not assumed. Silently averaging an empty list to 0.0 would record a
    # fabricated "the model produces zero tokens/sec" in results/efficiency.csv
    # instead of an honest "this CLI build didn't report it."
    if not tok_s_vals:
        print("WARNING: no tok/s parsed from llama-mtmd-cli output on this build "
              "-- recording tok_s as nan rather than a fake 0.0")
    _row(args.tag, args.host, runner=Path(args.model).name,
         size_gb=size_gb, mmproj_gb=mmproj_gb,
         peak_ram_gb=peak_ram_gb or float("nan"),
         tok_s=(sum(tok_s_vals) / len(tok_s_vals)) if tok_s_vals else float("nan"),
         e2e_s=sum(e for e, _ in warm) / len(warm),
         e2e_cold_s=runs[0][0])


def bench_hf(args: argparse.Namespace) -> None:
    import torch
    from ledgerlens.runners.hf_vlm import HFVLMRunner

    runner = HFVLMRunner(args.model, load_in_4bit=args.load_in_4bit)
    ckpt = Path(args.model)
    size_gb = (sum(f.stat().st_size for f in ckpt.rglob("*.safetensors")) / 1e9
               if ckpt.exists() else float("nan"))
    torch.cuda.reset_peak_memory_stats()
    times, toks = [], []
    for i in range(args.repeats):
        t0 = time.perf_counter()
        out = runner.generate(Path(args.image), PROMPT)
        dt = time.perf_counter() - t0
        n_tok = len(runner.processor.tokenizer(out)["input_ids"])
        times.append(dt)
        toks.append(n_tok / dt)
        print(f"run {i + 1}/{args.repeats}: e2e={dt:.2f}s ~tok/s={n_tok / dt:.1f}")
    warm_t, warm_k = (times[1:] or times), (toks[1:] or toks)
    _row(args.tag, args.host, runner=args.model, size_gb=size_gb,
         peak_ram_gb=torch.cuda.max_memory_allocated() / 1e9,
         tok_s=sum(warm_k) / len(warm_k), e2e_s=sum(warm_t) / len(warm_t),
         e2e_cold_s=times[0])


def bench_manual(args: argparse.Namespace) -> None:
    """On-phone numbers measured externally (app timer, adb dumpsys meminfo)."""
    _row(args.tag, args.host, runner="manual", size_gb=args.size_gb,
         peak_ram_gb=args.peak_ram_gb, tok_s=args.tok_s, e2e_s=args.e2e_s,
         notes=args.notes or "")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)

    g = sub.add_parser("gguf")
    g.add_argument("--model", required=True)
    g.add_argument("--mmproj", required=True)
    g.add_argument("--image", required=True)
    g.add_argument("--cli", help="path to llama-mtmd-cli")
    g.add_argument("--max-tokens", type=int, default=512)
    g.add_argument("--repeats", type=int, default=4)
    g.add_argument("--tag", required=True)
    g.add_argument("--host", required=True)
    g.set_defaults(func=bench_gguf)

    h = sub.add_parser("hf")
    h.add_argument("--model", required=True)
    h.add_argument("--image", required=True)
    h.add_argument("--load-in-4bit", action="store_true")
    h.add_argument("--repeats", type=int, default=4)
    h.add_argument("--tag", required=True)
    h.add_argument("--host", required=True)
    h.set_defaults(func=bench_hf)

    m = sub.add_parser("manual")
    m.add_argument("--tag", required=True)
    m.add_argument("--host", required=True)
    m.add_argument("--size-gb", type=float, required=True)
    m.add_argument("--peak-ram-gb", type=float, required=True)
    m.add_argument("--tok-s", type=float, required=True)
    m.add_argument("--e2e-s", type=float, required=True)
    m.add_argument("--notes")
    m.set_defaults(func=bench_manual)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
