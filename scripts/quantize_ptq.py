"""Weekend 3: PTQ points on the grid — AWQ int4 and GPTQ int8/int4 on the student LLM.

Both methods quantize the language model; the vision tower stays fp16 (it is small
relative to the LLM and quantizing it disproportionately hurts doc OCR fidelity —
this is itself a finding to report).

Calibration uses real task data: rendered prompts from a labeled train split, so
activation statistics match deployment.

    uv run --extra gpu python scripts/quantize_ptq.py awq \
        --model runs/sft-2b-teacher-agree-merged --out models/student2b-awq-int4
    uv run --extra gpu python scripts/quantize_ptq.py gptq --bits 8 \
        --model runs/sft-2b-teacher-agree-merged --out models/student2b-gptq-int8

NOTE: AutoAWQ/GPTQModel support for Qwen2-VL moves quickly; pin versions on the GPU
box and record them in results (the harness row's tag should encode method+bits).
The GGUF k-quant path (export_gguf.py) is the *deployment* int4; AWQ/GPTQ rows
exist to compare PTQ algorithms at equal bit-width on GPU.
"""

from __future__ import annotations

import argparse
import json


def calibration_texts(splits: list[str], n: int) -> list[str]:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from ledgerlens.schema import PROMPT

    texts = []
    for path in splits:
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                texts.append(PROMPT + "\n" + json.dumps(rec["gold"], ensure_ascii=False))
    return texts[:n]


def run_awq(args: argparse.Namespace) -> None:
    from awq import AutoAWQForCausalLM
    from transformers import AutoProcessor, AutoTokenizer

    quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": args.bits, "version": "GEMM"}
    model = AutoAWQForCausalLM.from_pretrained(args.model, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model.quantize(
        tokenizer,
        quant_config=quant_config,
        calib_data=calibration_texts(args.calib, args.n_calib) if args.calib else "pileval",
    )
    model.save_quantized(args.out)
    tokenizer.save_pretrained(args.out)
    AutoProcessor.from_pretrained(args.model).save_pretrained(args.out)
    print(f"AWQ int{args.bits} -> {args.out}")


def run_gptq(args: argparse.Namespace) -> None:
    from gptqmodel import GPTQModel, QuantizeConfig  # pip install gptqmodel on the GPU box

    cfg = QuantizeConfig(bits=args.bits, group_size=128)
    model = GPTQModel.load(args.model, cfg)
    model.quantize(calibration_texts(args.calib, args.n_calib) if args.calib else None)
    model.save(args.out)
    print(f"GPTQ int{args.bits} -> {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="method", required=True)
    for name, fn, default_bits in (("awq", run_awq, 4), ("gptq", run_gptq, 4)):
        p = sub.add_parser(name)
        p.add_argument("--model", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--bits", type=int, default=default_bits, choices=[4, 8])
        p.add_argument("--calib", nargs="*", help="unified jsonl split(s) for calibration")
        p.add_argument("--n-calib", type=int, default=256)
        p.set_defaults(func=fn)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
