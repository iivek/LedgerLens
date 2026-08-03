# LedgerLens

**Offline, on-device invoice & receipt extraction.** Photo of a receipt in → structured JSON out (vendor, date, currency, total, tax, line items) — running privately on a phone, no network, no cloud. Because it must run on a mid-range Android device, model compression (distillation → quantization → pruning) is a *requirement of the use case*, and the headline deliverable is the accuracy-vs-size/latency **Pareto grid**.

> **Status: Weekend 1 baselines only — no distillation/compression has happened yet.** These are stock, off-the-shelf Qwen2-VL-2B numbers, measured before any of *this project's* training. Every number below comes straight from `results/results.csv`, produced by `uv run ledgerlens eval` — nothing here is hand-computed.

## Baseline: stock Qwen2-VL-2B, zero fine-tuning

Measured on this repo's own eval harness, running the deployment runtime (llama.cpp, this exact GGUF) locally on a MacBook M1 Pro — not a rented GPU, not the phone yet. `fields_em_mean` averages only the fields each dataset actually annotates (SROIE: vendor/date/total; CORD: total/tax/line-items — see [CLAUDE.md](CLAUDE.md) on why).

| Variant | Split (n) | JSON valid | Field EM (mean) | Line-item F1 | Latency p50 / mean |
|---|---|---|---|---|---|
| Q4_K_M (940 MB) | SROIE (347) | 96.0% | 86.6% | — | 13.6s / 17.0s |
| Q4_K_M (940 MB) | CORD (100) | 98.0% | 60.0% | 76.0% | 8.9s / 13.7s |
| Q8_0 (1.5 GB) | SROIE (347) | 95.7% | 84.8% | — | 13.7s / 18.5s |
| Q8_0 (1.5 GB) | CORD (100) | 98.0% | 63.5% | 74.7% | 10.3s / 15.9s |

**Early read:** the stock 2B model is already a strong baseline — SROIE mean field exact-match in the mid-80s with no fine-tuning at all. Q8_0 is *not* meaningfully more accurate than Q4_K_M here (within noise, arguably slightly lower on SROIE) despite being 60% larger and slower — an early hint of diminishing PTQ returns worth checking against the AWQ/GPTQ grid later. CORD scores lower because it's a harder distribution (worse for zero-shot vendor/date-free extraction, richer line-item structure). Per the spec's contingency: if distillation doesn't clearly beat this stock baseline, the story pivots to "how small can we go," not "look how much we improved."

None of this involves the teacher (Qwen2.5-VL-7B), SFT, PTQ/QAT beyond stock GGUF quantization, or pruning yet — those are Weekends 2–3.

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
