#!/usr/bin/env bash
# gen_urls_tranco.sh — Generate benchmarks/fixtures/urls_tranco.tsv.
#
# Downloads the latest Tranco top-domain list (CC-BY, no auth required),
# then generates an equal number of matched-segment-structure synthetic
# negative domains.
#
# Usage:
#   bash benchmarks/fixtures/gen_urls_tranco.sh
#
# Optional env overrides:
#   GEN_N=200000           Number of positives (default 200k)
#   GEN_SEED=42            RNG seed (default 42)
#   TRANCO_CSV=<path>      Path to a pre-downloaded Tranco CSV (skips network)
#   TRANCO_CACHE=<path>    Save downloaded CSV here for future re-use

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_TSV="${SCRIPT_DIR}/urls_tranco.tsv"
N="${GEN_N:-200000}"
SEED="${GEN_SEED:-42}"

EXTRA_ARGS=()
if [[ -n "${TRANCO_CSV:-}" ]]; then
    EXTRA_ARGS+=("--cached-csv" "${TRANCO_CSV}")
fi
if [[ -n "${TRANCO_CACHE:-}" ]]; then
    EXTRA_ARGS+=("--cache-to" "${TRANCO_CACHE}")
fi

python3 "${SCRIPT_DIR}/gen_urls_tranco.py" \
    --out   "${OUT_TSV}" \
    --n     "${N}"       \
    --seed  "${SEED}"    \
    "${EXTRA_ARGS[@]}"

echo "[gen_urls_tranco.sh] Done. Output: ${OUT_TSV}"
