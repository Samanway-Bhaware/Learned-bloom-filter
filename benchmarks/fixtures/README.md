# Benchmark Fixtures

This directory contains deterministically generated fixture files used by the
Phase 5 benchmark suite.  The `.tsv` files are *not* committed to the
repository (see `.gitignore`) and must be regenerated before running
benchmarks:

```bash
bash benchmarks/fixtures/gen_words_en.sh
```

---

## `words_en.tsv`

### Purpose

A supervised membership dataset for evaluating whether a Learned Bloom Filter
can exploit character-level structure in real English words to achieve lower
false-positive rates than a classical Bloom Filter at matched memory budgets.

### Positive class (label = 1)

Source: `/usr/share/dict/words` (macOS / Linux system dictionary, ~235k entries).

Filter applied:
- Alphabetic characters only (`str.isalpha()` — no hyphens, apostrophes, digits)
- Length 3–20 characters (removes single-letter abbreviations and pathologically
  long medical terms)
- Lowercased and deduplicated

Result: ≈ 235k usable words.  Up to 200k are selected after a deterministic
shuffle (seed-based), so the same words are always members across re-runs.

### Negative class (label = 0)

**Design goal:** negatives should be *plausible* strings — same character
alphabet, same length distribution — so the model cannot cheat by exploiting
a trivially visible difference (e.g. length, non-alpha characters).

Procedure:
1. Compute the per-length histogram of the positive set.
2. For each length `l`, draw `count[l]` random strings uniformly from
   `[a-z]^l` using a fixed seed.
3. Discard any generated string that appears in the positive set (collision
   probability ≈ 1/26^l — negligible for l ≥ 3).

This is strictly harder for the model than using random character sequences
with a non-matched length distribution, and it prevents the trivial "English
words are longer" leakage.

### Why this is learnable (AUC >> 0.5)

English words are *not* uniformly distributed over [a-z]^l: they exhibit
strong morphological constraints (common prefixes `un-`, `re-`; suffixes
`-ing`, `-tion`, `-ed`; high-frequency digrams `th`, `he`, `in`; forbidden
letter combinations etc.).  Random strings drawn uniformly from [a-z]^l lack
these constraints.  A character n-gram classifier easily separates the two
classes.

This contrasts with the **previous broken dataset** (Step 3, committed
`464f940`) where positives and negatives were both drawn from the same
dictionary — making them exchangeable with respect to n-gram features and
yielding AUC ≈ 0.5.

### AUC validation

Before running C++ benchmarks, validate the dataset quality:

```bash
python3 scripts/validate_auc.py
# Expected output: Validation AUC = 0.XX  (should be >= 0.95)
# Exit 0 if AUC >= 0.80, exit 1 otherwise.
```

The script trains a sklearn `CountVectorizer(char, 3-5gram)` +
`LogisticRegression` on 80% of the data and evaluates ROC-AUC on the
remaining 20%.  This mirrors our C++ `LogisticRegressionModel` with
`NGramConfig{min_n=3, max_n=5}`.

### Generation command

```bash
# Default: 200k positives, seed=42
bash benchmarks/fixtures/gen_words_en.sh

# Custom size / seed
GEN_N=100000 GEN_SEED=7 bash benchmarks/fixtures/gen_words_en.sh
```

### File format

```
<word>\t<label>
```

- Positives (label `1`) appear first in the file.
- Non-members (label `0`) appear after.
- `bench_dataset.hpp::load_tsv()` relies on this ordering for its early-exit
  loading logic.
