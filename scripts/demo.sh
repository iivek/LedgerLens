#!/usr/bin/env bash
# LedgerLens demo: receipt photo in, JSON out, fully offline after a one-time download.
#
#   scripts/demo.sh path/to/receipt.jpg
#
# Needs llama.cpp's `llama-mtmd-cli` on PATH (macOS: `brew install llama.cpp`;
# elsewhere: https://github.com/ggml-org/llama.cpp/releases). Downloads the two GGUF
# files (~2.3 GB) into models/gguf/ on first run, then never touches the network again.
#
# The prompt below is byte-identical to `PROMPT` in src/ledgerlens/schema.py; a unit
# test (tests/test_demo_prompt.py) enforces that. Do not edit it here alone.
set -euo pipefail

HF_REPO="${LEDGERLENS_HF_REPO:-ii42/LedgerLens-Qwen2-VL-2B-Receipts-GGUF}"
MODEL_FILE="sft2b-teacher-agree-q4_k_m.gguf"
MMPROJ_FILE="mmproj-sft2b-teacher-agree-f16.gguf"

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model_dir="${LEDGERLENS_MODEL_DIR:-$here/models/gguf}"

PROMPT='Extract the following fields from this receipt or invoice image and answer with a single JSON object, nothing else:
{"vendor": string|null, "date": "YYYY-MM-DD"|null, "currency": ISO-4217 string|null, "total": number|null, "tax": number|null, "line_items": [{"description": string, "qty": number, "unit_price": number, "total": number}]}
Use null for fields not present. Do not invent values.'

usage() { sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'; exit 1; }
[ $# -eq 1 ] || usage
image="$1"
[ -f "$image" ] || { echo "demo.sh: image not found: $image" >&2; exit 1; }

command -v llama-mtmd-cli >/dev/null 2>&1 || {
  echo "demo.sh: llama-mtmd-cli not found on PATH." >&2
  echo "  macOS:  brew install llama.cpp" >&2
  echo "  other:  https://github.com/ggml-org/llama.cpp/releases (pick a build with mtmd)" >&2
  exit 1
}

fetch() {  # fetch <filename>
  local f="$1" dest="$model_dir/$1"
  [ -s "$dest" ] && return 0
  mkdir -p "$model_dir"
  echo "demo.sh: downloading $f from https://huggingface.co/$HF_REPO ..." >&2
  curl -fL --retry 3 -C - -o "$dest.part" "https://huggingface.co/$HF_REPO/resolve/main/$f"
  mv "$dest.part" "$dest"
}
fetch "$MODEL_FILE"
fetch "$MMPROJ_FILE"

# --temp 0 for deterministic output; -n caps the answer length; 2>/dev/null hides
# llama.cpp's load/perf logging so stdout is just the model's JSON.
exec llama-mtmd-cli \
  -m "$model_dir/$MODEL_FILE" \
  --mmproj "$model_dir/$MMPROJ_FILE" \
  --image "$image" \
  --temp 0 -n 512 \
  -p "$PROMPT" 2>/dev/null
