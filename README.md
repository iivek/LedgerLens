# LedgerLens

**Offline, on-device invoice & receipt extraction.** Photo of a receipt in → structured JSON out (vendor, date, currency, total, tax, line items) — running privately on a phone, no network, no cloud. Because it must run on a mid-range Android device, model compression (distillation → quantization → pruning) is a *requirement of the use case*, and the headline deliverable is the accuracy-vs-size/latency **Pareto grid**.

> **Status: Weekend 1 baselines only — no distillation/compression has happened yet.** These are stock, off-the-shelf Qwen2-VL-2B numbers, measured before any of *this project's* training. Every number below comes straight from `results/results.csv`, produced by `uv run ledgerlens eval` — nothing here is hand-computed.

## Baseline: stock Qwen2-VL-2B, zero fine-tuning

Measured on this repo's own eval harness, running the deployment runtime (llama.cpp, this exact GGUF) locally on a MacBook M1 Pro — not a rented GPU, not the phone yet. `fields_em_mean` averages only the fields each dataset actually annotates (SROIE: vendor/date/total; CORD: total/tax/line-items — see [CLAUDE.md](CLAUDE.md) on why).

*(Rescored — see `scripts/rescore.py` — after two metric bugs were found and fixed: a line-item null-handling bug that understated line-item F1 by ~9-10pp, and a date-normalization gap that let the model's non-ISO output coincidentally match SROIE gold's own non-canonical fallback text, inflating date accuracy. Original numbers are still in `results/results.csv` under the non-`-rescored` tags for the audit trail.)*

| Variant | Split (n) | JSON valid | Field EM (mean) | Vendor / Date / Total EM (SROIE) | Line-item F1 | Latency p50 / mean |
|---|---|---|---|---|---|---|
| Q4_K_M (940 MB) | SROIE (347) | 96.0% | 77.3% | 81.8% / 59.9% / 90.2% | — | 13.6s / 17.0s |
| Q4_K_M (940 MB) | CORD (100) | 98.0% | 60.0% | — | 86.0% | 8.9s / 13.7s |
| Q8_0 (1.5 GB) | SROIE (347) | 95.7% | 77.9% | 82.4% / 63.1% / 88.2% | — | 13.7s / 18.5s |
| Q8_0 (1.5 GB) | CORD (100) | 98.0% | 63.5% | — | 83.7% | 10.3s / 15.9s |

**Honest read, not a one-sided one:** Q8_0 is not more accurate than Q4_K_M in any way that survives scrutiny — and it isn't uniformly *worse* either. Per-field, the two trade places: Q8 leads on CORD total EM (+7pp) and SROIE vendor EM (+0.6pp); Q4 leads on SROIE date EM (+3.2pp) and SROIE total EM (+2pp). A real quantization effect shouldn't flip direction between datasets — this pattern is consistent with sampling noise at this sample size (SROIE n=347, CORD n=100; a single unpaired run each), not a "diminishing returns" finding. Treat it as unresolved, not as evidence, until it's checked with paired significance tests and repeated runs.

Date EM is the field most worth reading carefully: after the metric fix, it's the clear accuracy floor (~60-63%) despite total EM staying high (~88-90%) — largely because the model rarely follows the prompt's requested `YYYY-MM-DD` format even when it reads the date correctly, and the corrected metric no longer gives credit for that.

CORD scores lower on tax/total than SROIE's headline fields because it's a harder distribution (richer line-item structure, no vendor/date/currency annotated at all). Per the spec's contingency: if distillation doesn't clearly beat this stock baseline, the story pivots to "how small can we go," not "look how much we improved."

### SROIE train/test overlap — vendor-disjoint numbers

The Kaggle SROIE mirror (`urbikn/sroie-datasetv2`) is a re-split of the same underlying 973 receipts, not the official ICDAR test set. Checked directly against the current data: **7/347 test images are byte-identical to a training image** (left as-is — 2% of the split, not worth the churn to hunt down and relabel by hand), and **72.6% of test receipts share a vendor string with a training receipt** (252/347). Vendor overlap is the one worth correcting for, since it can inflate vendor accuracy independent of whether the model can actually read a receipt it hasn't seen the store's layout for before.

`scripts/rescore.py --vendor-disjoint-against <train-split>` restricts scoring to the 95 test receipts whose vendor does *not* appear in `sroie_train` — no re-running the model, just re-scoring the existing predictions:

| Variant | Vendor EM (full test, n=347) | Vendor EM (vendor-disjoint, n=95) | Date EM (disjoint) | Total EM (disjoint) |
|---|---|---|---|---|
| Q4_K_M | 81.8% | **61.1%** | 56.8% | 90.5% |
| Q8_0 | 82.4% | **62.1%** | 57.9% | 88.4% |

A 20-point gap. Important nuance: **this is the stock, zero-fine-tuned model** — it has never trained on any SROIE data, so this isn't training-set memorization in the usual sense. The more likely explanation is that vendors repeating across train/test tend to be larger, more common chains, and the model's own general-purpose pretraining gives it an edge recognizing familiar branding — an edge that doesn't extend to rarer, less-common vendors. Either way, the full-test vendor number is not a reliable estimate of performance on a vendor the model hasn't effectively seen before, whatever the cause. **This check becomes essential once real fine-tuning happens on `sroie_train`** — at that point the same gap could additionally reflect genuine training-set memorization, and vendor-disjoint scoring is how that gets caught rather than published as a clean win.

None of this involves the teacher (Qwen2.5-VL-7B), SFT, PTQ/QAT beyond stock GGUF quantization, or pruning yet — those are Weekends 2–3. **The vision tower (mmproj, 1.33 GB) has not been quantized in either variant above and is larger than the 4-bit LLM (0.99 GB)** — the 940MB/1.5GB figures describe the LLM only; deployed footprint including mmproj is 2.32 GB (Q4) vs 2.98 GB (Q8), a much smaller ratio than the LLM-only numbers suggest. This is also the leading candidate explanation for why LLM-only quantization shows so little effect above.

**On latency vs. the spec's "~1-2 seconds on a phone" target:** these numbers (13-18s mean, up to 40-49s p95) are already far above that, and they're on a **GPU-accelerated M1 Pro** — the current Android JNI code runs **CPU-only**. A mid-range Android phone's CPU is meaningfully slower than an M1 Pro's GPU, so on-device latency today is very likely in the tens of seconds to low minutes, not 1-2 seconds. The compression plan as designed (int4 LLM + pruning) targets a ~1.5-2x speedup, which does not close a gap of this size — most of the latency is plausibly vision-token prefill (the mmproj is unquantized and image-token budget isn't yet a compression axis). The 1-2s target needs to be either restated honestly or the plan needs to add image-resolution/token-budget reduction and GPU delegation as explicit levers, before this claim goes in front of anyone evaluating compression work.

## Architecture

```
CORD + SROIE + synthetic invoices
        │
        ▼
Qwen2.5-VL-7B teacher ──labels──▶ SFT (LoRA) ──▶ Qwen2-VL-2B student
                                                    │
                              ┌─────────────────────┼──────────────────┐
                              ▼                     ▼                  ▼
                         PTQ (AWQ/GPTQ)        QAT (torchao)      pruning (+recovery)
                              └─────────────────────┼──────────────────┘
                                                    ▼
                                        GGUF (q4_k_m) + mmproj
                                                    ▼
                                 llama.cpp / mtmd — laptop CPU & Android app
```

Every variant is one row in `results/results.csv` (accuracy, from the harness) joined with `results/efficiency.csv` (size / peak RAM / tok/s / latency) on `tag` — one point on the Pareto grid.

## Setup

```sh
uv sync                     # core + dev (harness, metrics, synthetic gen)
uv sync --extra data        # + HF datasets (CORD download)
uv sync --extra render && uv run playwright install chromium   # optional; system Chrome also works
# GPU box only:
uv sync --extra gpu
```

## The one-command eval

```sh
uv run ledgerlens eval --runner hf --model Qwen/Qwen2-VL-2B-Instruct \
    --split data/processed/cord_test.jsonl --tag student2b-stock --host gpu
```

Appends one aggregate row to `results/results.csv` and persists per-sample raw outputs to `runs/<tag>/predictions.jsonl` for audit. The `llamacpp` runner evaluates the *deployed* GGUF through a local `llama-server`, i.e. the same runtime the phone uses:

```sh
llama-server -m models/gguf/ledgerlens-q4_k_m.gguf --mmproj models/gguf/mmproj-ledgerlens-f16.gguf -c 4096 --port 8080
uv run ledgerlens eval --runner llamacpp --split data/processed/cord_test.jsonl \
    --tag student2b-q4km --host laptop-cpu
```

**Metrics** (fixed before any training): per-field exact-match & F1 (vendor, date, currency, total, tax; totals compared with separator normalization and ±0.005 tolerance), JSON-validity rate, line-item F1 with Hungarian matching over (description, qty, price), and latency percentiles.

## Data

```sh
uv run ledgerlens convert cord              # CORD-v2 via HuggingFace → data/processed/
uv run ledgerlens convert sroie             # needs SROIE2019/ under data/raw/ (see src/ledgerlens/data/sroie.py)
uv run ledgerlens synth --n 2000            # synthetic invoices → data/synthetic/
```

The synthetic generator renders Jinja2 HTML templates (thermal receipt + A4 invoice), with randomized vendors, currencies (incl. `1.234,56`-style separators and zero-decimal IDR/VND), then applies phone-photo augmentations (perspective, shadows, low light, motion blur, JPEG). Gold labels are exact by construction.

CORD and SROIE are **research-use** datasets; model weights trained on them are labeled accordingly. See [DATA_LICENSES.md](DATA_LICENSES.md) for exact terms — CORD is confirmed CC-BY-4.0, SROIE's dataset-level license could not be independently verified and is treated as research-use only pending confirmation from the ICDAR/RRC organizers.

**Personal holdout** (spec §4 — the honest "does it actually work" set, ~50 real receipts, never trained on): drop photos in `data/personal_holdout/raw/`, run `uv run ledgerlens convert personal` once to scaffold a blank label JSON per photo in `data/personal_holdout/labels/` (matching the `Extraction` schema), hand-fill each with what the receipt actually says, then rerun the same command to build `data/personal_holdout.jsonl`. A typo'd field name fails loudly (not silently null) since these are hand-typed. Never committed (`data/personal_holdout/` is gitignored); redact or exclude before publishing anything derived from it.

## Training & compression (GPU box, 24 GB)

```sh
python scripts/teacher_label.py --split data/processed/cord_train.jsonl --out data/distill/cord_train_teacher.jsonl
python scripts/train_sft.py --data data/distill/*.jsonl --policy teacher-agree --out runs/sft-2b --merge
python scripts/quantize_ptq.py awq  --model runs/sft-2b-merged --out models/student2b-awq-int4
python scripts/qat_torchao.py       --model runs/sft-2b-merged --data data/distill/cord_train_teacher.jsonl --out models/student2b-qat
python scripts/prune.py             --model runs/sft-2b-merged --drop 6 --calib data/distill/cord_train_teacher.jsonl --out models/student2b-prune6
python scripts/export_gguf.py       --model runs/sft-2b-merged --name ledgerlens
python scripts/bench_efficiency.py gguf --model models/gguf/ledgerlens-q4_k_m.gguf --mmproj models/gguf/mmproj-ledgerlens-f16.gguf --image <receipt.jpg> --tag student2b-q4km --host laptop-cpu
```

## Android app

`android/` contains a Kotlin/Compose app (camera → local llama.cpp `mtmd` inference → JSON, visible latency timer, **no INTERNET permission**). See [android/README.md](android/README.md) — it requires a llama.cpp submodule checkout and Android Studio; the JNI layer is written against the current mtmd API and must be validated against a pinned llama.cpp tag on first build.

## Tests

```sh
uv run pytest
```

## License

Apache-2.0 (code). Datasets and derived weights carry their own research-use terms.
