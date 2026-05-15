#!/usr/bin/env python3
"""Generate benchmarks/fixtures/urls_tranco.tsv.

Positive class  — top-N domains from the Tranco research-grade ranking.
                  Tranco combines Alexa, Cisco Umbrella, Majestic Million,
                  and Farsight into a stable, reproducible weekly list.
                  Source: https://tranco-list.eu  (CC-BY, no auth required)

Negative class  — synthetic domain-like strings with MATCHED structure:
                  each domain "a.b.c" → negative has same segment lengths
                  filled with random lowercase alphanumeric characters, so
                  the n-gram model cannot exploit length or segment-count
                  differences.  Filtered to guarantee zero overlap with positives.

Label format:  <domain>TAB<label>  (positives first, label=1; negatives after, label=0)

Usage:
    # Fetch from Tranco API (requires internet), cache raw CSV, then generate:
    python3 benchmarks/fixtures/gen_urls_tranco.py \\
        --out benchmarks/fixtures/urls_tranco.tsv \\
        --n 200000 --seed 42

    # Re-use a previously cached Tranco CSV:
    python3 benchmarks/fixtures/gen_urls_tranco.py \\
        --cached-csv /tmp/tranco.csv \\
        --out benchmarks/fixtures/urls_tranco.tsv \\
        --n 200000 --seed 42
"""

import argparse
import csv
import io
import random
import string
import sys
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Tranco API constants
# ---------------------------------------------------------------------------
_API_BASE = "https://tranco-list.eu"
_LIST_LATEST_API = _API_BASE + "/api/lists/date/latest"
_DOWNLOAD_TMPL = _API_BASE + "/download/{list_id}/full"
_USER_AGENT = "lbf-benchmark-fixture/1.0 (github research; see benchmarks/fixtures/README.md)"

_MAX_ATTEMPTS_MULT = 200


def _fetch_latest_list_id() -> str:
    """Query the Tranco API for the most recent list identifier."""
    import json

    req = urllib.request.Request(_LIST_LATEST_API, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode())
    # API returns {"list_id": "XXXXX", ...}
    list_id = data.get("list_id") or data.get("listid") or data.get("id")
    if not list_id:
        raise RuntimeError(f"Unexpected API response: {data}")
    return str(list_id)


def _download_tranco(list_id: str) -> str:
    """Download the full Tranco list CSV as a string."""
    url = _DOWNLOAD_TMPL.format(list_id=list_id)
    print(f"[gen_urls_tranco] Downloading list {list_id} from {url} ...", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as resp:
        raw = resp.read()
    # Tranco may gzip-encode even without Accept-Encoding; handle if needed.
    if raw[:2] == b"\x1f\x8b":
        import gzip
        raw = gzip.decompress(raw)
    return raw.decode("utf-8", errors="replace")


def load_positives(csv_text: str, max_n: int, seed: int) -> list[str]:
    """Parse Tranco CSV and return up to max_n clean domain strings."""
    domains: list[str] = []
    seen: set[str] = set()
    reader = csv.reader(io.StringIO(csv_text))
    for row in reader:
        if len(row) < 2:
            continue
        domain = row[1].strip().lower()
        if not domain or domain in seen:
            continue
        # Keep only: lowercase alphanum, dots, hyphens; exclude IPs and empties.
        if all(c in string.ascii_lowercase + string.digits + ".-" for c in domain):
            if "." in domain and len(domain) >= 4:
                domains.append(domain)
                seen.add(domain)
    # Deterministic shuffle so the positive selection is reproducible.
    rng = random.Random(seed ^ 0xDEAD_BEEF)
    rng.shuffle(domains)
    return domains[:max_n]


def _synthetic_domain(parts: list[str], rng: random.Random) -> str:
    """
    Generate one synthetic domain matching the segment-length structure of parts.
    Each dot-separated segment is replaced with a random lowercase alphanum string
    of the same length.  This preserves dot count and segment lengths — the only
    distinguishing features available to a random adversary.
    """
    alphanum = string.ascii_lowercase + string.digits
    synthetic_parts = ["".join(rng.choices(alphanum, k=len(p))) for p in parts]
    return ".".join(synthetic_parts)


def generate_negatives(positives: list[str], seed: int) -> list[str]:
    """
    Generate one negative per positive using matched-segment-length synthetic domains.
    Filters out any accidental match with the positive set.
    """
    pos_set: frozenset[str] = frozenset(positives)
    rng = random.Random(seed)

    negatives: list[str] = []
    for domain in positives:
        parts = domain.split(".")
        max_attempts = _MAX_ATTEMPTS_MULT
        generated = False
        for _ in range(max_attempts):
            candidate = _synthetic_domain(parts, rng)
            if candidate not in pos_set:
                negatives.append(candidate)
                generated = True
                break
        if not generated:
            # Extremely unlikely; fall back to longer random string
            negatives.append("x" + "".join(rng.choices(string.ascii_lowercase, k=len(domain))))
    return negatives


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate urls_tranco.tsv with matched-structure synthetic negatives."
    )
    ap.add_argument("--out", required=True, help="Output TSV path")
    ap.add_argument(
        "--n",
        type=int,
        default=200_000,
        help="Max positives to include (default 200k; capped by Tranco list size)",
    )
    ap.add_argument(
        "--cached-csv",
        default=None,
        help="Path to a previously downloaded Tranco CSV (skips network fetch)",
    )
    ap.add_argument(
        "--cache-to",
        default=None,
        help="If fetching, save the raw CSV here for future re-use",
    )
    ap.add_argument("--seed", type=int, default=42, help="RNG seed (default 42)")
    args = ap.parse_args()

    # ---------------------------------------------------------------- fetch
    if args.cached_csv:
        print(f"[gen_urls_tranco] Reading cached CSV from {args.cached_csv} ...")
        csv_text = Path(args.cached_csv).read_text(encoding="utf-8", errors="replace")
    else:
        try:
            list_id = _fetch_latest_list_id()
            print(f"[gen_urls_tranco] Latest Tranco list id: {list_id}")
            csv_text = _download_tranco(list_id)
            if args.cache_to:
                Path(args.cache_to).write_text(csv_text, encoding="utf-8")
                print(f"[gen_urls_tranco] Cached raw CSV → {args.cache_to}")
        except Exception as exc:
            print(
                f"[gen_urls_tranco] ERROR fetching Tranco: {exc}\n"
                "  Use --cached-csv to supply a local file, or check your connection.",
                file=sys.stderr,
            )
            sys.exit(1)

    # --------------------------------------------------------------- parse
    print(f"[gen_urls_tranco] Parsing CSV, selecting up to {args.n} positives ...")
    positives = load_positives(csv_text, args.n, args.seed)
    print(f"[gen_urls_tranco] {len(positives)} positives loaded.")

    # -------------------------------------------------------------- negate
    print(
        f"[gen_urls_tranco] Generating {len(positives)} matched-structure negatives "
        f"(seed={args.seed}) ..."
    )
    negatives = generate_negatives(positives, args.seed)
    print(f"[gen_urls_tranco] {len(negatives)} negatives generated.")

    # --------------------------------------------------------- sanity check
    overlap = frozenset(positives) & frozenset(negatives)
    if overlap:
        print(
            f"[gen_urls_tranco] ERROR: {len(overlap)} overlap(s)! Report a bug.",
            file=sys.stderr,
        )
        sys.exit(1)
    print("[gen_urls_tranco] Overlap check: PASS (0 collisions).")

    # ---------------------------------------------------------------- write
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for d in positives:
            f.write(f"{d}\t1\n")
        for d in negatives:
            f.write(f"{d}\t0\n")
    print(
        f"[gen_urls_tranco] Wrote {len(positives)} positives + {len(negatives)} negatives"
        f" → {out_path}"
    )


if __name__ == "__main__":
    main()
