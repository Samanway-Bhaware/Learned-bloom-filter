#!/usr/bin/env python3
"""Generate a uniform random 64-bit integer dataset for LBF benchmarks.

Output format: TSV with one entry per line:
    <hex_uint64>\t<label>
where label is 1 for members and 0 for non-members.  Members come first in
the file so benchmark code can slice any prefix up to --n for the member set.

Disjointness guarantee: members and non-members are drawn from separate RNG
streams and the non-member set is explicitly filtered against the member set.
For N=1 000 000 and 64-bit output, expected collisions ≈ N²/(2·2⁶⁴) < 0.03.

Usage:
    python3 gen_uniform_int.py [--n 1000000] [--n-probes 200000] [--seed 42] \\
                               [--out uniform_int.tsv]
"""
import argparse
import random
import sys
import time


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--n", type=int, default=1_000_000,
        help="Number of member keys (label=1). Default: 1 000 000."
    )
    parser.add_argument(
        "--n-probes", type=int, default=200_000,
        help="Number of non-member probe keys (label=0). Default: 200 000."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="MT19937 seed for reproducibility. Default: 42."
    )
    parser.add_argument(
        "--out", default="uniform_int.tsv",
        help="Output file path. Default: uniform_int.tsv."
    )
    args = parser.parse_args()

    t0 = time.perf_counter()

    # --- members ---
    rng_m = random.Random(args.seed)
    members_list = [rng_m.getrandbits(64) for _ in range(args.n)]
    member_set = set(members_list)

    # --- non-members (filtered against member set) ---
    rng_p = random.Random(args.seed + 0xDEAD_BEEF)
    probes: list[int] = []
    while len(probes) < args.n_probes:
        v = rng_p.getrandbits(64)
        if v not in member_set:
            probes.append(v)

    # --- write ---
    with open(args.out, "w", newline="\n") as fh:
        for v in members_list:
            fh.write(f"{v:016x}\t1\n")
        for v in probes:
            fh.write(f"{v:016x}\t0\n")

    elapsed = time.perf_counter() - t0
    print(
        f"[gen_uniform_int] {len(members_list):,} members + {len(probes):,} probes "
        f"written to '{args.out}' in {elapsed:.1f}s",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
