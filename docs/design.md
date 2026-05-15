# Design Document — Learned Bloom Filter Library (lbf)

> Status: **skeleton** — filled in progressively through Phases 2–6.

---

## 1. Problem Statement and Motivation

A classical Bloom filter answers set-membership queries in O(1) time and
sub-linear space, at the cost of a tunable false-positive rate (FPR). The
false-negative rate is always zero by construction.

Kraska et al. (2018) observed that when the key distribution is learnable
(i.e., positives and negatives are distinguishable by some model), a small
classifier can replace or augment the Bloom filter, reducing memory usage
while preserving the zero-false-negative guarantee via a backup BF.

This library demonstrates that tradeoff concretely, with reproducible
benchmarks that show both when the learned variant wins and when it does not.

---

## 2. Architecture

```
                         ┌─────────────────────────┐
                         │   LearnedBloomFilter<K>  │
                         │                          │
   query(key)            │  1. model.predict(key)   │
──────────────────►      │     >= tau? → true        │
                         │                          │
                         │  2. backup_bf.contains() │
                         └───────────┬──────────────┘
                                     │
                    ┌────────────────┴───────────────┐
                    │                                │
            ┌───────▼──────┐              ┌──────────▼──────┐
            │ MembershipModel│              │  BloomFilter<K>  │
            │ (LogReg/ONNX) │              │  (backup, small) │
            └───────────────┘              └──────────────────┘
```

*(Full Mermaid diagram added in Phase 6.)*

---

## 3. Key Design Decisions

### 3.1 Hash Function: xxHash3 vs MurmurHash3

*Decision:* **xxHash3** (128-bit variant).

*Rationale:* xxHash3 is measurably faster than MurmurHash3 on modern CPUs
(typically 2x on AVX2-capable hardware), is header-only, BSD licensed, and
actively maintained (used in production by zstd). Its 128-bit output maps
directly to the two independent hash values required by the
Kirsch-Mitzenmacher double-hashing optimization, avoiding a second hash call.
MurmurHash3 is more commonly cited in Bloom filter literature but has been
unmaintained since ~2016. We vendor xxHash as a single header via FetchContent.

### 3.2 Model Choice

*Decision:* **Logistic Regression** as the primary C++ implementation.

*Rationale:* TBD in Phase 3.

### 3.3 Threshold Selection

*Decision:* TBD in Phase 4. Options include grid search on a validation set
or setting tau to minimize total memory (backup BF bits + model bits).

### 3.4 Memory Layout

Bit array stored as `std::vector<uint64_t>`, word-level operations. Avoids
`std::vector<bool>` (proxy-reference issues) and `std::bitset` (fixed size).

---

## 4. Tradeoffs and Limitations

- LBF requires a training phase; classical BF does not. In insert-heavy
  streaming scenarios the classical BF will win.
- The model inference cost per query can exceed the hash cost for simple key
  distributions. See benchmarks/benchmarks.md.
- ONNX Runtime backend increases binary size and adds a runtime dependency.

---

## 5. References

1. Kraska, T. et al. "The Case for Learned Index Structures." SIGMOD 2018.
2. Mitzenmacher, M. "A Model for Learned Bloom Filters and Optimizing by
   Sandwiching." NeurIPS 2018.
3. Kirsch, A. and Mitzenmacher, M. "Less Hashing, Same Performance: Building
   a Better Bloom Filter." ESA 2006.
