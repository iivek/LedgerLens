"""Weekend 3: PTQ points on the grid — AWQ int4 and GPTQ int8/int4 on the student LLM.

Both methods quantize the language model. The vision tower (mmproj) is NOT
quantized by this script or anywhere else in the current pipeline. Measure
its share of deployed size/latency (results/results.csv: the vision tower
GGUF is larger than the 4-bit LLM GGUF) before treating that as settled --
it isn't yet an established finding, just an open question this script
doesn't answer.

    uv run --extra gpu python scripts/quantize_ptq.py awq \
        --model runs/sft-2b-teacher-agree-merged --out models/student2b-awq-int4
    uv run --extra gpu python scripts/quantize_ptq.py gptq --bits 8 \
        --model runs/sft-2b-teacher-agree-merged --out models/student2b-gptq-int8

CALIBRATION IS CURRENTLY TEXT-ONLY -- A REAL LIMITATION, NOT YET FIXED:
calibration_texts() below builds calibration samples from PROMPT + gold JSON
text, with no image ever passed through. At deployment the LLM's activations
are dominated by ~500-1500 image-embedding tokens with a very different
magnitude distribution than plain text; AWQ's per-channel scales (and GPTQ's
Hessian estimate) computed from text-only calibration may not reflect the
activations the quantized model actually sees. This was flagged by
independent review and is NOT fixed here, because fixing it correctly
requires confirming AutoAWQ/GPTQModel's calibration API actually accepts
multimodal (image+text) samples for this specific model class -- something
that needs verifying against the library versions on the GPU box, not a
blind code change from an environment with no GPU to test it against. Before
trusting AWQ/GPTQ results: check whether these libraries expose an image-
aware calibration path for Qwen2-VL, and if not, treat this as a known
methodology gap in any writeup rather than silently reporting the numbers.

NOTE: AutoAWQ/GPTQModel support for Qwen2-VL moves quickly; pin versions on the GPU
box and record them in results (the harness row's tag should encode method+bits).
The GGUF k-quant path (export_gguf.py) is the *deployment* int4; AWQ/GPTQ rows
exist to compare PTQ algorithms at equal bit-width on GPU.
"""

from __future__ import annotations

import argparse
import json


def calibration_texts(splits: list[str], n: int, seed: int = 0) -> list[str]:
    import random
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
    # Shuffle before truncating to n: slicing an unshuffled concatenation would
    # make the whole calibration set whichever split was listed first if it
    # alone has >= n samples (e.g. --calib cord.jsonl sroie.jsonl with n=256
    # and cord having 800 samples -> 100% CORD, 0% SROIE calibration data).
    random.Random(seed).shuffle(texts)
    return texts[:n]


def run_awq(args: argparse.Namespace) -> None:
    # VERIFY FIRST RUN: Qwen's official Qwen2-VL AWQ recipe uses a custom
    # quantizer subclass for multimodal calibration, not this generic
    # AutoAWQForCausalLM.from_pretrained + text-only calib_data path. Confirm
    # this actually quantizes only the LLM submodule correctly (check the saved
    # checkpoint's vision tower is untouched fp16) before trusting the output —
    # do not assume this generic path is correct for a VLM checkpoint.
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
