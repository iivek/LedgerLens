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
uv run ledgerlens convert cord|sroie|personal
uv run ledgerlens synth --n 2000
```

GPU-side scripts (`scripts/`) need `--extra gpu` and real GPU compute this Mac doesn't have — don't try to run them locally. In practice they run on **Kaggle's free GPU tier** (T4; `NvidiaTeslaP100` is incompatible with modern PyTorch wheels — always set `machine_shape: "NvidiaTeslaT4"` in kernel-metadata.json), driven by chunked kernel pushes since one Kaggle session can't fit a full multi-hour labeling/training job. Order: `teacher_label.py` → `train_sft.py` → `quantize_ptq.py` / `qat_torchao.py` / `prune.py` → `export_gguf.py` → `bench_efficiency.py`.

This machine has multiple named Kaggle accounts under `~/.kaggle/tokens/` for quota pooling. **Always use `~/.kaggle/kaggle-as <account> <args>`, never bare `kaggle`** — it verifies identity before running and restores the prior credential after, so it can't silently run as (or clobber) another project's account on this shared machine. This repo's account is recorded in `.kaggle-account` (`johannvekstr`, with `ioannesio` as an automatic quota-exhaustion failover — see the chunked-driver pattern below). A `kaggle` CLI call failing with a permission/identity error is very likely a credential swap by another project sharing this machine, not a real problem with your kernel/dataset — recheck with `kaggle-as` before assuming something broke.

## Architecture: the contracts that hold everything together

- **`src/ledgerlens/schema.py` is frozen.** `Extraction` (pydantic) + `PROMPT` are the single contract shared by the harness, teacher labeling, SFT targets, and the Android app. `android/.../Prompts.kt` embeds a byte-identical copy of `PROMPT` inside the Qwen ChatML wrapper — if one changes, the other must, but after baselines are run neither should (it invalidates all comparisons).
- **Normalization before comparison** (`normalize.py`): predictions and gold both pass through separator-tolerant number parsing, ambiguity-safe date canonicalization, and currency aliasing. Metric changes belong there, not in `metrics.py`.
- **Runners return raw text, not parsed JSON** (`runners/`): JSON validity is itself a metric, so parsing failures are counted in the harness, never swallowed. The `llamacpp` runner talks to `llama-server` so laptop/phone rows measure the same runtime that ships. `mock` exists so the whole pipeline runs modelless in tests.
- **Unified data format** (`data/unified.py`): every source (CORD, SROIE, synthetic, personal holdout) converts to one JSONL-plus-images layout; downstream code never knows the source. CORD lacks vendor/date/currency (null); SROIE lacks line items/tax — they complement each other.
  - **Known caveat, found by visual inspection, not fixed:** ~13% of SROIE training images (81/626, checked directly) are full-page office scans (>15MP, up to 35MP) with large blank margins and occasional handwritten filing numbers — not phone photos, a different capture modality than CORD's mostly hand-held shots. Since MIN_PIXELS/MAX_PIXELS caps *total* pixel budget regardless of source resolution, these get downscaled to the same budget as a tightly-cropped image, leaving the actual receipt text proportionally smaller/blurrier within that budget. Left as-is (no auto-crop implemented) rather than engineering a fix without GPU access to validate it helps — but if SROIE accuracy on the eval set is unexpectedly weak after training, check whether it's concentrated in this large-scan subset before concluding the model just "isn't good at SROIE."
- **Metrics score only what a source annotates** (`metrics.py:ANNOTATED_FIELDS`): a sample's `source` (`cord`/`sroie`/`synthetic`/`personal`) determines which fields are scored — e.g. SROIE gold never has line items, so a model that correctly extracts them isn't penalized for it. `harness.py` passes `source` through automatically. Extending to a new source requires adding its annotated-field set here first.
  - **Known caveat, deliberately not "fixed":** this masking is per-*source* (CORD annotates tax as a field-type), not per-*sample*. A CORD receipt with gold `tax: null` could mean either "genuinely no tax line" or "the annotator skipped it" — there's no per-sample signal in the dataset to tell which, so a model correctly reading a real tax line the annotator missed scores 0 there, and a model that outputs null everywhere gets free both-null credit. CORD's ~55% tax EM partly reflects this, not just reading accuracy. Not fixable without better upstream annotations; report null vs. non-null gold subsets separately if this matters for a specific claim, rather than trusting the blended number as pure accuracy.
- **Synthetic augmentation has a severity budget** (`data/synthetic/augment.py:SEVERITY_BUDGET`): each of the five phone-photo degradations draws its own severity and is scaled down once cumulative severity exceeds the budget, so realistic single/double effects stay strong but five-at-once never renders text fully illegible (this was a real bug — early unrestrained stacking produced unreadable training images).
- **Results are append-only CSVs keyed by `tag`**: `results/results.csv` (accuracy, from the harness) joins `results/efficiency.csv` (size/RAM/tok/s/latency, from `bench_efficiency.py`) on `tag`. One tag = one Pareto point. Per-sample raw outputs persist in `runs/<tag>/predictions.jsonl` for audit.
- **Training reuses eval code**: `train_sft.py:build_messages`/`make_collator` are imported by `qat_torchao.py`; teacher labels carry both teacher output and gold plus an agreement flag so blend policy (`gold` | `teacher` | `teacher-agree`) is chosen at train time without re-labeling. `build_messages`'s heavy imports are all deferred into `main()`, so the blend-policy logic itself is unit-testable without torch (`tests/test_train_sft.py`) — keep it that way.
- **Personal holdout has no source of truth to convert** (`data/personal.py`): unlike CORD/SROIE/synthetic, a human has to supply gold labels for real photographed receipts. First run scaffolds blank label JSON per photo; second run (after hand-filling) builds the split. Unknown/typo'd label keys raise loudly — deliberately stricter than `Extraction`'s default leniency (which stays lenient for parsing model output, where tolerating stray fields is correct).
- **Android** (`android/`): Compose UI + JNI over llama.cpp's `mtmd`. Built without an SDK present — correct-by-inspection only; the JNI/mtmd API surface is the expected first build failure (see android/README.md troubleshooting). No INTERNET permission by design; keep it that way.

## Hard rules (spec §10 — non-negotiable)

- Every published number must come from the harness/bench scripts, reproducibly. Never hand-write results into README or writeups.
- Never imply production deployment or real users. Label laptop-CPU demos honestly as such.
- Dataset licenses (`DATA_LICENSES.md`) are verified against primary sources, never assumed from a search-engine summary or an adjacent code repo's license file (a real near-miss: a downloader script's MIT license is not the dataset's license). SROIE's dataset-level license is unconfirmed and stays flagged research-use-only until verified with the RRC/ICDAR organizers.

## Environment notes

- macOS, no GPU, no Android SDK. Locally runnable: tests, synthetic generation (uses system Chrome or Playwright), CORD conversion, mock-runner evals.
- Large artifacts (`data/`, `models/`, `runs/`) are gitignored; never commit datasets or weights.
