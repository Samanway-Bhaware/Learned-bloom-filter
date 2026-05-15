#!/usr/bin/env python3
"""Validate words_en.tsv dataset quality via held-out AUC.

Trains a character n-gram (3-5) logistic regression on 80% of the data
and evaluates ROC-AUC on the remaining 20%.  This mirrors the C++
LogisticRegressionModel / NGramConfig used in the benchmark binary.

If AUC < --min-auc (default 0.80) the script exits with code 1 and
prints a diagnostic.  The C++ benchmark suite should not be run on a
dataset that fails this gate.

Usage:
    python3 scripts/validate_auc.py
    python3 scripts/validate_auc.py --tsv path/to/words_en.tsv --min-auc 0.85

Dependencies:
    pip install scikit-learn numpy
"""

import argparse
import sys


def main() -> None:
    try:
        import numpy as np
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import train_test_split
    except ImportError:
        print(
            "ERROR: scikit-learn / numpy not installed.\n"
            "  pip install scikit-learn numpy",
            file=sys.stderr,
        )
        sys.exit(2)

    ap = argparse.ArgumentParser(
        description="AUC gate for words_en.tsv before running C++ benchmarks."
    )
    ap.add_argument(
        "--tsv",
        default="benchmarks/fixtures/words_en.tsv",
        help="Path to TSV dataset (default: benchmarks/fixtures/words_en.tsv)",
    )
    ap.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Held-out validation fraction (default 0.20)",
    )
    ap.add_argument(
        "--min-auc",
        type=float,
        default=0.80,
        help="Minimum acceptable AUC (default 0.80; exit 1 if below)",
    )
    ap.add_argument(
        "--max-features",
        type=int,
        default=16384,
        help="CountVectorizer max_features (matches C++ feature_dim, default 16384)",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # ------------------------------------------------------------------ load
    print(f"[validate_auc] Loading {args.tsv} ...", flush=True)
    words: list[str] = []
    labels: list[int] = []
    with open(args.tsv, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            tab = line.rfind("\t")
            if tab == -1:
                continue
            words.append(line[:tab])
            labels.append(int(line[tab + 1 :]))

    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    print(
        f"[validate_auc] {len(words)} samples: {n_pos} positives, {n_neg} negatives."
    )
    if n_pos == 0 or n_neg == 0:
        print("ERROR: dataset has only one class — cannot compute AUC.", file=sys.stderr)
        sys.exit(2)

    # ------------------------------------------------------------------ split
    X_tr, X_val, y_tr, y_val = train_test_split(
        words,
        labels,
        test_size=args.val_split,
        random_state=args.seed,
        stratify=labels,
    )
    print(
        f"[validate_auc] Train={len(X_tr)}, Val={len(X_val)} "
        f"(val_split={args.val_split:.0%}, seed={args.seed})"
    )

    # ------------------------------------------------------------------ fit
    print(
        f"[validate_auc] Fitting char n-gram (3,5) LogReg "
        f"(max_features={args.max_features}) ...",
        flush=True,
    )
    vec = CountVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        max_features=args.max_features,
        binary=True,  # presence, not count — matches our C++ hasher
    )
    X_tr_v = vec.fit_transform(X_tr)
    X_val_v = vec.transform(X_val)

    clf = LogisticRegression(
        C=10.0,          # ~l2_reg=1e-4 at our scale
        max_iter=300,
        random_state=args.seed,
        solver="lbfgs",
        n_jobs=-1,
    )
    clf.fit(X_tr_v, y_tr)

    # ------------------------------------------------------------------ AUC
    y_prob = clf.predict_proba(X_val_v)[:, 1]
    auc = float(roc_auc_score(y_val, y_prob))

    # Threshold-specific accuracy at τ ∈ {0.3, 0.5, 0.7}
    print(f"\n  Validation ROC-AUC : {auc:.4f}")
    print("  Threshold | FPR (val) | TPR (val)")
    print("  ----------+-----------+----------")
    for tau in [0.3, 0.5, 0.7]:
        y_pred = (y_prob >= tau).astype(int)
        tp = int(np.sum((y_pred == 1) & (np.array(y_val) == 1)))
        fp = int(np.sum((y_pred == 1) & (np.array(y_val) == 0)))
        fn = int(np.sum((y_pred == 0) & (np.array(y_val) == 1)))
        tn = int(np.sum((y_pred == 0) & (np.array(y_val) == 0)))
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        print(f"  τ={tau:.1f}      | {fpr:.4f}     | {tpr:.4f}")
    print()

    # ------------------------------------------------------------------ gate
    if auc >= args.min_auc:
        print(
            f"[validate_auc] AUC {auc:.4f} >= {args.min_auc:.2f}: PASS — "
            "proceed to C++ benchmark."
        )
        sys.exit(0)
    else:
        print(
            f"[validate_auc] AUC {auc:.4f} < {args.min_auc:.2f}: FAIL — "
            "investigate feature extraction or dataset before benchmarking.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
