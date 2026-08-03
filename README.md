# LedgerLens

**Offline, on-device invoice & receipt extraction.** Photo of a receipt in → structured JSON out (vendor, date, currency, total, tax, line items) — running privately on a phone, no network, no cloud. Because it must run on a mid-range Android device, model compression (distillation → quantization → pruning) is a *requirement of the use case*, and the headline deliverable is the accuracy-vs-size/latency **Pareto grid**.

> **Status: pre-training scaffold.** The eval harness, data pipeline, synthetic generator, training/compression scripts and Android app skeleton are built; no models have been trained yet, so there are **no results to report**. The baseline-vs-final table will appear here once produced by the harness — every number in this README will come from `results/results.csv`, reproducibly.

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
