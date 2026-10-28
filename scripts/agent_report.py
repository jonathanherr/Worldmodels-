"""Summarize scripts.agent_squares results: best side per (n, mode) vs grid, store and records.

    python -m scripts.agent_report /mnt/project-files/squares/agents/run1.json [...]
"""

import json
import sys
from collections import defaultdict


def main():
    rows = []
    for p in sys.argv[1:]:
        rows += json.load(open(p))
    best = defaultdict(dict)
    ref = {}
    for r in rows:
        if not r["valid"]:
            continue
        k = (r["n"], r["mode"])
        best[r["n"]][r["mode"]] = min(best[r["n"]].get(r["mode"], 1e9), r["official"])
        ref[r["n"]] = (r["grid"], r["store"], r["record"])
    modes = sorted({m for d in best.values() for m in d})
    print("n     grid   store  record  " + "  ".join(f"{m:>9s}" for m in modes))
    beat_grid = defaultdict(int)
    frac = defaultdict(list)
    wins = []
    for n in sorted(best):
        g, s, rec = ref[n]
        cells = []
        for m in modes:
            v = best[n].get(m)
            cells.append(f"{v:9.4f}" if v else "        -")
            if v is None:
                continue
            if v < g - 1e-6:
                beat_grid[m] += 1
            if g - rec > 1e-6:
                frac[m].append((g - v) / (g - rec))
            if v < s - 1e-9:
                wins.append((n, m, v, s))
        print(f"{n:4d} {g:6.0f} {s:7.4f} {rec:7.4f}  " + "  ".join(cells))
    print()
    beatable = sum(1 for n in best if ref[n][0] - ref[n][2] > 1e-6)
    for m in modes:
        f = frac[m]
        print(f"{m:9s} beats grid at {beat_grid[m]:3d}/{beatable} beatable n; "
              f"mean share of grid-to-record gap closed {sum(f) / max(len(f), 1):+.3f}")
    print("store wins:", wins or "none")


if __name__ == "__main__":
    main()
