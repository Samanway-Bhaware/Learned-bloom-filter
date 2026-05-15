#!/usr/bin/env python3
"""Generate benchmark plots from Google Benchmark JSON output.

Reads results/*.json and produces four plots in results/plots/:
  1. throughput.png      — query ops/sec classical vs LBF (miss path, 100k)
  2. fpr_comparison.png  — measured FPR with Wilson 95% CI (100k, all τ)
  3. memory_pareto.png   — FPR vs memory scatter (Pareto frontier)
  4. latency_pct.png     — p50/p99 latency bar chart (miss path, 100k)

Usage:
    python3.13 scripts/plot_benchmarks.py [--results-dir results/]
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
except ImportError:
    print("ERROR: matplotlib / numpy not installed.\n  pip install matplotlib numpy",
          file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})
COLORS = {
    "classical": "#2196F3",   # blue
    "lbf_t03":   "#FF9800",   # orange  τ=0.3
    "lbf_t05":   "#4CAF50",   # green   τ=0.5
    "lbf_t07":   "#9C27B0",   # purple  τ=0.7
}

# ---------------------------------------------------------------------------
# JSON loading helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _means(data: dict) -> list[dict]:
    """Return only _mean rows from a GBench JSON."""
    return [b for b in data.get("benchmarks", []) if b["name"].endswith("_mean")]


def _c(bmark: dict, key: str, default=0.0) -> float:
    return float(bmark.get(key, default))


def _find(rows: list[dict], substr: str) -> dict | None:
    for r in rows:
        if substr in r["name"]:
            return r
    return None


def _find_all(rows: list[dict], substr: str) -> list[dict]:
    return [r for r in rows if substr in r["name"]]


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------

def extract_classical(rows: list[dict], dataset: str, n: str) -> dict | None:
    """FPR, memory, latency for a classical BF miss-path benchmark."""
    tag = f"Contains_Miss/{dataset}_{n}"
    r = _find(rows, tag)
    if r is None:
        return None
    return {
        "fpr":       _c(r, "lbf_fpr_measured"),
        "fpr_lo":    _c(r, "lbf_fpr_wilson_lo95"),
        "fpr_hi":    _c(r, "lbf_fpr_wilson_hi95"),
        "mem_kb":    _c(r, "lbf_memory_bytes") / 1024,
        "lat_ns":    r["cpu_time"],
        "p50":       _c(r, "lbf_p50_ns"),
        "p99":       _c(r, "lbf_p99_ns"),
        "throughput": _c(r, "items_per_second") / 1e6,  # M ops/s
    }


def extract_lbf(rows: list[dict], dataset: str, n: str, tau_tag: str) -> dict | None:
    """FPR, memory, latency for an LBF miss-path benchmark."""
    tag = f"Contains_Miss/{dataset}_{n}_{tau_tag}"
    r = _find(rows, tag)
    if r is None:
        return None
    return {
        "fpr":            _c(r, "lbf_fpr_measured"),
        "fpr_lo":         _c(r, "lbf_fpr_wilson_lo95"),
        "fpr_hi":         _c(r, "lbf_fpr_wilson_hi95"),
        "mem_kb":         _c(r, "lbf_total_memory_bytes") / 1024,
        "model_kb":       _c(r, "lbf_model_memory_bytes") / 1024,
        "backup_kb":      _c(r, "lbf_backup_memory_bytes") / 1024,
        "lat_ns":         r["cpu_time"],
        "p50":            _c(r, "lbf_p50_ns"),
        "p99":            _c(r, "lbf_p99_ns"),
        "throughput":     _c(r, "items_per_second") / 1e6,
        "fastpath_frac":  _c(r, "lbf_fastpath_frac"),
        "backup_count":   _c(r, "lbf_backup_count"),
        "threshold":      _c(r, "lbf_threshold"),
    }


# ---------------------------------------------------------------------------
# Plot 1 — Throughput comparison (miss path, N=100k)
# ---------------------------------------------------------------------------

def plot_throughput(rows_classical: list[dict], rows_lbf: list[dict],
                    out: Path) -> None:
    datasets   = ["uniform_int", "words_en", "urls_tranco"]
    ds_labels  = ["Uniform int", "English words", "Tranco URLs"]

    classical_thr = []
    lbf_thr_t05   = []
    for ds, dsn in zip(datasets, ds_labels):
        c = extract_classical(rows_classical, ds, "100k")
        classical_thr.append(c["throughput"] if c else 0.0)
        l = extract_lbf(rows_lbf, ds, "100k", "t05")
        lbf_thr_t05.append(l["throughput"] if l else 0.0)

    x = np.arange(len(datasets))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w/2, classical_thr, w, label="Classical BF", color=COLORS["classical"])
    ax.bar(x + w/2, lbf_thr_t05,   w, label="LBF (τ=0.5)", color=COLORS["lbf_t05"])
    ax.set_xticks(x)
    ax.set_xticklabels(ds_labels)
    ax.set_ylabel("Throughput (M ops/s, miss path)")
    ax.set_title("Query Throughput — Classical BF vs LBF (N=100k)")
    ax.legend()
    # Annotate with slowdown factors
    for i, (c, l) in enumerate(zip(classical_thr, lbf_thr_t05)):
        if c > 0 and l > 0:
            slowdown = c / l
            ax.text(i + w/2, l + 0.05, f"{slowdown:.1f}×\nslower",
                    ha="center", va="bottom", fontsize=8, color="#555")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"[plot] {out}")


# ---------------------------------------------------------------------------
# Plot 2 — FPR comparison with Wilson CI (N=100k, all τ)
# ---------------------------------------------------------------------------

def plot_fpr_comparison(rows_classical: list[dict], rows_lbf: list[dict],
                        out: Path) -> None:
    datasets  = ["words_en", "urls_tranco"]
    ds_labels = ["English words", "Tranco URLs"]
    taus      = [("t03", 0.3), ("t05", 0.5), ("t07", 0.7)]
    tau_colors = [COLORS["lbf_t03"], COLORS["lbf_t05"], COLORS["lbf_t07"]]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=False)
    for ax, ds, dslabel in zip(axes, datasets, ds_labels):
        c = extract_classical(rows_classical, ds, "100k")
        fpr_c = c["fpr"] * 100 if c else None
        lo_c  = (c["fpr"] - c["fpr_lo"]) * 100 if c else 0
        hi_c  = (c["fpr_hi"] - c["fpr"]) * 100 if c else 0

        names, fprs, lo_errs, hi_errs, colors = [], [], [], [], []
        if fpr_c is not None:
            names.append("Classical BF")
            fprs.append(fpr_c)
            lo_errs.append(lo_c)
            hi_errs.append(hi_c)
            colors.append(COLORS["classical"])

        for (tag, tau), col in zip(taus, tau_colors):
            l = extract_lbf(rows_lbf, ds, "100k", tag)
            if l:
                names.append(f"LBF τ={tau}")
                fprs.append(l["fpr"] * 100)
                lo_errs.append((l["fpr"] - l["fpr_lo"]) * 100)
                hi_errs.append((l["fpr_hi"] - l["fpr"]) * 100)
                colors.append(col)

        x = np.arange(len(names))
        bars = ax.bar(x, fprs, color=colors,
                      yerr=[lo_errs, hi_errs], capsize=4, error_kw={"elinewidth": 1.5})
        ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", label="1% target")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15, ha="right", fontsize=9)
        ax.set_ylabel("Measured FPR (%)")
        ax.set_title(f"{dslabel} — FPR (N=100k, Wilson 95% CI)")
        ax.legend(fontsize=8)
        for bar, fpr in zip(bars, fprs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(hi_errs) * 0.05,
                    f"{fpr:.2f}%", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"[plot] {out}")


# ---------------------------------------------------------------------------
# Plot 3 — Memory vs FPR Pareto scatter (log-log)
# ---------------------------------------------------------------------------

def _pareto_panel(ax, ds: str, dslabel: str,
                  rows_classical: list[dict], rows_lbf: list[dict]) -> None:
    """Draw one Pareto panel (log-log) onto ax."""
    taus    = [("t03", 0.3), ("t05", 0.5), ("t07", 0.7)]
    mk      = "o"

    # Theoretical classical BF curve: m = N * ln(1/p) / ln(2)^2 bytes
    n = 100_000
    p_range = np.logspace(np.log10(0.003), np.log10(0.20), 300)
    classical_mem_kb = n * np.log(1.0 / p_range) / (np.log(2) ** 2) / 8 / 1024
    ax.plot(p_range * 100, classical_mem_kb, "--", color=COLORS["classical"],
            linewidth=1.5, label="Classical BF (theoretical)")

    # Classical measured point
    c = extract_classical(rows_classical, ds, "100k")
    if c:
        ax.scatter([c["fpr"] * 100], [c["mem_kb"]],
                   marker=mk, s=110, color=COLORS["classical"],
                   label="Classical BF (measured)", zorder=5)

    # LBF points
    tau_cols = [COLORS["lbf_t03"], COLORS["lbf_t05"], COLORS["lbf_t07"]]
    for (tag, tau), col in zip(taus, tau_cols):
        l = extract_lbf(rows_lbf, ds, "100k", tag)
        if l:
            ax.scatter([l["fpr"] * 100], [l["mem_kb"]],
                       marker="^", s=110, color=col, zorder=5,
                       label=f"LBF τ={tau}")
            ax.errorbar([l["fpr"] * 100], [l["mem_kb"]],
                        xerr=[[(l["fpr"] - l["fpr_lo"]) * 100],
                               [(l["fpr_hi"] - l["fpr"]) * 100]],
                        fmt="none", color=col, capsize=3, linewidth=1)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Measured FPR (%, log scale)")
    ax.set_ylabel("Memory (KB, log scale)")
    ax.set_title(f"{dslabel} — Memory vs FPR Pareto (N=100k, log-log)")
    ax.legend(fontsize=8, loc="upper right")
    ax.annotate("← lower-left is better", xy=(0.03, 0.05), xycoords="axes fraction",
                fontsize=8, color="#555")


def plot_memory_pareto(rows_classical: list[dict], rows_lbf: list[dict],
                       out: Path) -> None:
    """Combined Pareto scatter (both datasets, log-log axes)."""
    datasets  = ["words_en", "urls_tranco"]
    ds_labels = ["English words", "Tranco URLs"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, ds, dslabel in zip(axes, datasets, ds_labels):
        _pareto_panel(ax, ds, dslabel, rows_classical, rows_lbf)

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"[plot] {out}")


def plot_memory_pareto_single(rows_classical: list[dict], rows_lbf: list[dict],
                               ds: str, dslabel: str, out: Path) -> None:
    """Single-dataset Pareto scatter (log-log), for embedding in docs."""
    fig, ax = plt.subplots(figsize=(7, 5))
    _pareto_panel(ax, ds, dslabel, rows_classical, rows_lbf)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"[plot] {out}")


# ---------------------------------------------------------------------------
# Plot 4 — Latency percentiles p50 / p99 (miss path, N=100k)
# ---------------------------------------------------------------------------

def plot_latency_pct(rows_classical: list[dict], rows_lbf: list[dict],
                     out: Path) -> None:
    datasets  = ["words_en", "urls_tranco"]
    ds_labels = ["English words", "Tranco URLs"]
    taus      = [("t03", 0.3), ("t05", 0.5), ("t07", 0.7)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    for ax, ds, dslabel in zip(axes, datasets, ds_labels):
        names, p50s, p99s, colors = [], [], [], []

        c = extract_classical(rows_classical, ds, "100k")
        if c:
            names.append("Classical BF")
            p50s.append(c["p50"])
            p99s.append(c["p99"])
            colors.append(COLORS["classical"])

        tau_cols = [COLORS["lbf_t03"], COLORS["lbf_t05"], COLORS["lbf_t07"]]
        for (tag, tau), col in zip(taus, tau_cols):
            l = extract_lbf(rows_lbf, ds, "100k", tag)
            if l:
                names.append(f"LBF τ={tau}")
                p50s.append(l["p50"])
                p99s.append(l["p99"])
                colors.append(col)

        x = np.arange(len(names))
        w = 0.35
        b1 = ax.bar(x - w/2, p50s, w, color=colors, alpha=0.9, label="p50")
        b2 = ax.bar(x + w/2, p99s, w, color=colors, alpha=0.4, label="p99",
                    edgecolor=[c for c in colors], linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15, ha="right", fontsize=9)
        ax.set_ylabel("Latency (ns, miss path)")
        ax.set_title(f"{dslabel} — p50/p99 Latency (N=100k)")
        # Custom legend: solid=p50, hatched=p99
        p50_patch = mpatches.Patch(color="gray", alpha=0.9, label="p50")
        p99_patch = mpatches.Patch(color="gray", alpha=0.4, label="p99")
        ax.legend(handles=[p50_patch, p99_patch], fontsize=8)

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"[plot] {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Generate benchmark plots from GBench JSON.")
    ap.add_argument("--results-dir", default="results",
                    help="Directory containing GBench JSON files (default: results/)")
    args = ap.parse_args()

    rdir = Path(args.results_dir)
    pdir = rdir / "plots"
    pdir.mkdir(parents=True, exist_ok=True)

    # Load all JSON files and merge their benchmarks
    classical_rows: list[dict] = []
    lbf_rows:       list[dict] = []

    for jf in sorted(rdir.glob("classical_bf*.json")):
        classical_rows.extend(_means(_load_json(jf)))
        print(f"[plot] Loaded {jf} ({len(classical_rows)} classical means so far)")

    for jf in sorted(rdir.glob("learned_bf*.json")):
        rows = _means(_load_json(jf))
        lbf_rows.extend(rows)
        print(f"[plot] Loaded {jf} ({len(rows)} LBF means)")

    if not classical_rows:
        print("ERROR: no classical_bf*.json found in results/", file=sys.stderr)
        sys.exit(1)
    if not lbf_rows:
        print("ERROR: no learned_bf*.json found in results/", file=sys.stderr)
        sys.exit(1)

    print(f"[plot] Total: {len(classical_rows)} classical, {len(lbf_rows)} LBF mean rows")

    plot_throughput(classical_rows, lbf_rows, pdir / "throughput.png")
    plot_fpr_comparison(classical_rows, lbf_rows, pdir / "fpr_comparison.png")
    plot_memory_pareto(classical_rows, lbf_rows, pdir / "memory_pareto.png")
    plot_memory_pareto_single(classical_rows, lbf_rows,
                               "words_en", "English words",
                               pdir / "memory_pareto_words.png")
    plot_memory_pareto_single(classical_rows, lbf_rows,
                               "urls_tranco", "Tranco URLs",
                               pdir / "memory_pareto_urls.png")
    plot_latency_pct(classical_rows, lbf_rows, pdir / "latency_pct.png")

    print(f"\n[plot] All plots written to {pdir}/")


if __name__ == "__main__":
    main()
