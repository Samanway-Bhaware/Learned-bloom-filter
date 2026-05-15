#!/usr/bin/env bash
# gen_words_en.sh — Generate benchmarks/fixtures/words_en.tsv.
#
# Reads /usr/share/dict/words (macOS/Linux system dictionary).
# Output: words_en.tsv with up to 200k positives (real words) and
#         an equal number of matched-length synthetic negatives.
#
# Usage:
#   bash benchmarks/fixtures/gen_words_en.sh
#
# Optional env overrides:
#   GEN_N=200000   Number of positives (default 200k)
#   GEN_SEED=42    RNG seed (default 42)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_TSV="${SCRIPT_DIR}/words_en.tsv"
N="${GEN_N:-200000}"
SEED="${GEN_SEED:-42}"

python3 "${SCRIPT_DIR}/gen_words_en.py" \
    --out   "${OUT_TSV}" \
    --n     "${N}"       \
    --seed  "${SEED}"

echo "[gen_words_en.sh] Done. Output: ${OUT_TSV}"
