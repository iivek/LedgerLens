# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What LedgerLens is

An offline, on-device invoice/receipt extractor (photo → structured JSON) built as a model-compression portfolio project. Read [ledgerlens_project_spec.md](ledgerlens_project_spec.md) before substantive work — it defines scope, models, the weekend plan, and hard rules. The headline deliverable is the Pareto grid: accuracy vs bits/size/latency, one point per compression variant.

## Commands

```sh
uv sync                      # core env (harness, metrics, synthetic gen); --extra data|render|gpu as needed
uv run pytest                # test suite (fast, no models needed)
uv run pytest tests/test_metrics.py -k line_items   # single test
uv run ledgerlens eval --runner hf|llamacpp --split <jsonl> --tag <variant> --host <where>
uv run ledgerlens convert cord|sroie
uv run ledgerlens synth --n 2000
```

GPU-side scripts (`scripts/`) run on a rented 24 GB box with `--extra gpu`, in this order: `teacher_label.py` → `train_sft.py` → `quantize_ptq.py` / `qat_torchao.py` / `prune.py` → `export_gguf.py` → `bench_efficiency.py`. They cannot run on this Mac; don't try.

## Architecture: the contracts that hold everything together

- **`src/ledgerlens/schema.py` is frozen.** `Extraction` (pydantic) + `PROMPT` are the single contract shared by the harness, teacher labeling, SFT targets, and the Android app. `android/.../Prompts.kt` embeds a byte-identical copy of `PROMPT` inside the Qwen ChatML wrapper — if one changes, the other must, but after baselines are run neither should (it invalidates all comparisons).
- **Normalization before comparison** (`normalize.py`): predictions and gold both pass through separator-tolerant number parsing, ambiguity-safe date canonicalization, and currency aliasing. Metric changes belong there, not in `metrics.py`.
- **Runners return raw text, not parsed JSON** (`runners/`): JSON validity is itself a metric, so parsing failures are counted in the harness, never swallowed. The `llamacpp` runner talks to `llama-server` so laptop/phone rows measure the same runtime that ships. `mock` exists so the whole pipeline runs modelless in tests.
- **Unified data format** (`data/unified.py`): every source (CORD, SROIE, synthetic, personal holdout) converts to one JSONL-plus-images layout; downstream code never knows the source. CORD lacks vendor/date/currency (null); SROIE lacks line items/tax — they complement each other.
- **Results are append-only CSVs keyed by `tag`**: `results/results.csv` (accuracy, from the harness) joins `results/efficiency.csv` (size/RAM/tok/s/latency, from `bench_efficiency.py`) on `tag`. One tag = one Pareto point. Per-sample raw outputs persist in `runs/<tag>/predictions.jsonl` for audit.
- **Training reuses eval code**: `train_sft.py:build_messages`/`make_collator` are imported by `qat_torchao.py`; teacher labels carry both teacher output and gold plus an agreement flag so blend policy (`gold` | `teacher` | `teacher-agree`) is chosen at train time without re-labeling.
- **Android** (`android/`): Compose UI + JNI over llama.cpp's `mtmd`. Built without an SDK present — correct-by-inspection only; the JNI/mtmd API surface is the expected first build failure (see android/README.md troubleshooting). No INTERNET permission by design; keep it that way.

## Hard rules (spec §10 — non-negotiable)

- Every published number must come from the harness/bench scripts, reproducibly. Never hand-write results into README or writeups.
- Never imply production deployment or real users. Label laptop-CPU demos honestly as such.
- The README currently states "no results to report" — that stays until real harness rows exist.

## Environment notes

- macOS, no GPU, no Android SDK. Locally runnable: tests, synthetic generation (uses system Chrome or Playwright), CORD conversion, mock-runner evals.
- Large artifacts (`data/`, `models/`, `runs/`) are gitignored; never commit datasets or weights.
