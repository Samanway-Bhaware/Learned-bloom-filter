#!/usr/bin/env python3
"""Generate benchmarks/fixtures/words_en.tsv.

Positive class  — real English words from /usr/share/dict/words:
    filtered to alphabetic-only, length 3–20 chars, lowercased,
    deduplicated, then deterministically shuffled.

Negative class  — synthetic random alphabetic strings:
    length distribution is MATCHED to the positive set (same per-length
    count), so the model cannot exploit a length mismatch as a shortcut.
    Strings are filtered to guarantee no overlap with the positive set.

Label format: <word>TAB<label>   (positives first, label=1; negatives
after, label=0).  Matches the TSV convention used by bench_dataset.hpp.

Usage:
    python3 benchmarks/fixtures/gen_words_en.py \\
        --out benchmarks/fixtures/words_en.tsv \\
        --n 200000 --seed 42

See README.md in this directory for dataset design rationale.
"""

import argparse
import random
import string
import sys
from collections import Counter
from pathlib import Path

WORDS_FILE = "/usr/share/dict/words"
MIN_LEN = 3
MAX_LEN = 20
# Safety multiplier: try this many random strings per slot before giving up.
_MAX_ATTEMPTS_MULT = 200


def load_positives(words_file: str, max_n: int, seed: int) -> list[str]:
    """Return up to *max_n* usable words: alpha-only, len 3-20, lowercased."""
    words: list[str] = []
    seen: set[str] = set()
    with open(words_file, encoding="utf-8", errors="ignore") as f:
        for line in f:
            w = line.strip().lower()
            if w.isalpha() and MIN_LEN <= len(w) <= MAX_LEN and w not in seen:
                words.append(w)
                seen.add(w)
    # Deterministic shuffle so member/non-member split is reproducible.
    rng = random.Random(seed ^ 0xABCD_1234)
    rng.shuffle(words)
    return words[:max_n]


def generate_negatives(positives: list[str], seed: int) -> list[str]:
    """
    Generate one negative per positive using matched-length random strings.

    Strings are drawn uniformly from [a-z]^length.  Any string that
    appears in the positive set is discarded and re-drawn.  Collision
    probability is negligible (≈ |positives| / 26^length).
    """
    pos_set: frozenset[str] = frozenset(positives)
    length_counts: Counter[int] = Counter(len(w) for w in positives)
    alphabet: str = string.ascii_lowercase
    rng = random.Random(seed)

    negatives: list[str] = []
    for length in sorted(length_counts):
        needed = length_counts[length]
        max_attempts = needed * _MAX_ATTEMPTS_MULT
        generated = 0
        attempts = 0
        while generated < needed and attempts < max_attempts:
            s = "".join(rng.choices(alphabet, k=length))
            if s not in pos_set:
                negatives.append(s)
                generated += 1
            attempts += 1
        if generated < needed:
            print(
                f"[gen_words_en] WARNING: only generated {generated}/{needed} "
                f"negatives for length {length} after {attempts} attempts.",
                file=sys.stderr,
            )
    return negatives


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate words_en.tsv with matched-length synthetic negatives."
    )
    ap.add_argument("--out", required=True, help="Output TSV path")
    ap.add_argument(
        "--n",
        type=int,
        default=200_000,
        help="Max positives to include (default 200k; capped by dictionary size)",
    )
    ap.add_argument(
        "--words",
        default=WORDS_FILE,
        help=f"Source dictionary file (default {WORDS_FILE})",
    )
    ap.add_argument("--seed", type=int, default=42, help="RNG seed (default 42)")
    args = ap.parse_args()

    print(f"[gen_words_en] Loading positives from {args.words} ...")
    positives = load_positives(args.words, args.n, args.seed)
    print(f"[gen_words_en] {len(positives)} positives after filter/dedup/cap.")

    print(
        f"[gen_words_en] Generating {len(positives)} matched-length negatives "
        f"(seed={args.seed}) ..."
    )
    negatives = generate_negatives(positives, args.seed)
    print(f"[gen_words_en] {len(negatives)} negatives generated.")

    # Sanity: no overlap
    overlap = frozenset(positives) & frozenset(negatives)
    if overlap:
        print(
            f"[gen_words_en] ERROR: {len(overlap)} overlap(s) found!  "
            "Increase --seed or report a bug.",
            file=sys.stderr,
        )
        sys.exit(1)
    print("[gen_words_en] Overlap check: PASS (0 collisions).")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for w in positives:
            f.write(f"{w}\t1\n")
        for w in negatives:
            f.write(f"{w}\t0\n")

    print(
        f"[gen_words_en] Wrote {len(positives)} positives + {len(negatives)} negatives"
        f" → {out_path}"
    )


if __name__ == "__main__":
    main()
