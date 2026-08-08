"""`ledgerlens` CLI — one command per pipeline stage.

    ledgerlens eval --runner hf --model Qwen/Qwen2-VL-2B-Instruct \
        --split data/processed/cord_test.jsonl --tag student2b-fp16 --host gpu
    ledgerlens eval --runner llamacpp --url http://127.0.0.1:8080 \
        --model-path models/gguf/ledgerlens-q4_k_m.gguf --mmproj-path models/gguf/mmproj-f16.gguf \
        --split data/processed/cord_test.jsonl --tag student2b-q4km --host laptop-cpu
    ledgerlens convert cord --out data/processed
    ledgerlens synth --n 2000 --out data/synthetic
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _cmd_eval(args: argparse.Namespace) -> None:
    from .harness import run_eval
    from .runners.base import get_runner

    if args.runner == "hf":
        if not args.model:
            raise SystemExit("--model is required for --runner hf")
        runner = get_runner("hf", model_id=args.model, load_in_4bit=args.load_in_4bit)
    elif args.runner == "llamacpp":
        runner = get_runner("llamacpp", base_url=args.url, name=args.tag,
                           model_path=args.model_path, mmproj_path=args.mmproj_path)
    else:
        raise SystemExit(f"unknown runner {args.runner}")

    row = run_eval(runner, Path(args.split), tag=args.tag, host=args.host,
                   results_csv=Path(args.results), limit=args.limit)
    width = max(len(k) for k in row)
    for k, v in row.items():
        print(f"{k:<{width}}  {v}")


def _cmd_convert(args: argparse.Namespace) -> None:
    out = Path(args.out)
    if args.dataset == "cord":
        from .data.cord import convert_cord
        convert_cord(out)
    elif args.dataset == "sroie":
        from .data.sroie import convert_sroie
        convert_sroie(out)
    elif args.dataset == "personal":
        from .data.personal import convert_personal
        convert_personal(out)
    else:
        raise SystemExit(f"unknown dataset {args.dataset}")


def _cmd_synth(args: argparse.Namespace) -> None:
    from .data.synthetic.generate import generate_dataset
    generate_dataset(n=args.n, out_dir=Path(args.out), seed=args.seed,
                     augment=not args.no_augment)


def main() -> None:
    parser = argparse.ArgumentParser(prog="ledgerlens")
    sub = parser.add_subparsers(dest="command", required=True)

    p_eval = sub.add_parser("eval", help="run the eval harness, append a row to results CSV")
    p_eval.add_argument("--runner", choices=["hf", "llamacpp"], required=True)
    p_eval.add_argument("--model", help="HF model id or local checkpoint path (hf runner)")
    p_eval.add_argument("--url", default="http://127.0.0.1:8080", help="llama-server URL")
    p_eval.add_argument("--model-path", help="path to the GGUF llama-server is serving, "
                                             "for provenance (sha256 + size) in results.csv")
    p_eval.add_argument("--mmproj-path", help="path to the mmproj GGUF, for provenance")
    p_eval.add_argument("--load-in-4bit", action="store_true", help="bnb 4-bit (teacher on 24GB)")
    p_eval.add_argument("--split", required=True, help="unified jsonl split")
    p_eval.add_argument("--tag", required=True, help="variant name; one point on the Pareto grid")
    p_eval.add_argument("--host", required=True, help="where this ran: gpu | laptop-cpu | phone-<model>")
    p_eval.add_argument("--results", default="results/results.csv")
    p_eval.add_argument("--limit", type=int, help="evaluate only the first N samples")
    p_eval.set_defaults(func=_cmd_eval)

    p_conv = sub.add_parser("convert", help="convert a public dataset to the unified format")
    p_conv.add_argument("dataset", choices=["cord", "sroie", "personal"])
    p_conv.add_argument("--out", default="data/processed")
    p_conv.set_defaults(func=_cmd_convert)

    p_synth = sub.add_parser("synth", help="generate synthetic invoices")
    p_synth.add_argument("--n", type=int, default=2000)
    p_synth.add_argument("--out", default="data/synthetic")
    p_synth.add_argument("--seed", type=int, default=17)
    p_synth.add_argument("--no-augment", action="store_true")
    p_synth.set_defaults(func=_cmd_synth)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
