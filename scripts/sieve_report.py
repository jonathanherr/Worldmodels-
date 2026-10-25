"""Summarize a sieve_squares results file: per mode, how close each gets to the store.

    python -m scripts.sieve_report sieve_results.jsonl
"""

import collections
import json
import sys

import numpy as np


def main(path):
    rs = [json.loads(line) for line in open(path)]
    by_mode = collections.defaultdict(list)
    best = collections.defaultdict(dict)
    for r in rs:
        by_mode[r["mode"]].append(r)
        if r["L"]:
            b = best[r["n"]]
            b[r["mode"]] = min(b.get(r["mode"], 1e9), r["L"])
            b["store"] = r["best"]
    print("mode   runs  ok   median(L-store)  p10(L-store)  n-best-of-modes  wins")
    for m, v in sorted(by_mode.items()):
        gaps = np.array([r["L"] - r["best"] for r in v if r["L"]])
        top = sum(1 for b in best.values() if m in b and b[m] <= min(b.get(o, 1e9) for o in by_mode) + 1e-9)
        print(f"{m:6s} {len(v):5d} {len(gaps):4d}  {np.median(gaps):15.4f}  {np.percentile(gaps, 10):12.4f}"
              f"  {top:15d}  {(gaps < -1e-9).sum():4d}")
    print("\nn  store  " + "  ".join(sorted(by_mode)))
    for n in sorted(best):
        b = best[n]
        print(n, f"{b['store']:.4f}", "  ".join(f"{b.get(m, float('nan')):.4f}" for m in sorted(by_mode)))


if __name__ == "__main__":
    main(sys.argv[1])
