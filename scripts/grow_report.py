"""Summarize grow_squares.py results: gap closed per setting and per axis.

gap closed = (grid - side) / (grid - best_file) on n where the best file beats the grid
(1.0 = matches the best file, 0 = grid, negative = worse than grid). On grid-optimal n we count how
often a run reaches the grid side (within 1e-6).

  python scripts/grow_report.py grow_main.jsonl --best best_sides.json [--by k,adjacency]
"""

import argparse
import json
import math
from collections import defaultdict

import numpy as np

AXES = ["k", "adjacency", "direction", "rate", "pick", "band", "angles"]


def load(path, best):
    rows = []
    for line in open(path):
        r = json.loads(line)
        n = r["n"]
        grid = math.ceil(math.sqrt(n))
        b = best[n]
        r["grid_opt"] = b >= grid - 1e-9
        for key in ("direct", "best"):
            v = r[key]
            r[key + "_gap"] = None if r["grid_opt"] else (grid - v) / (grid - b)
        r["at_grid"] = r["best"] <= grid + 1e-6
        r["win"] = r["best"] < b - 1e-9
        rows.append(r)
    return rows


def table(rows, keys):
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r.get(k) for k in keys)].append(r)
    out = []
    for key, rs in groups.items():
        bg = [r["best_gap"] for r in rs if r["best_gap"] is not None]
        dg = [r["direct_gap"] for r in rs if r["direct_gap"] is not None]
        go = [r["at_grid"] for r in rs if r["grid_opt"]]
        # best over seeds per n, then mean over beatable n
        per_n = defaultdict(list)
        for r in rs:
            if r["best_gap"] is not None:
                per_n[r["n"]].append(r["best_gap"])
        out.append(dict(zip(keys, key), runs=len(rs), gap=np.mean(bg) if bg else float("nan"),
                        gap_bestseed=np.mean([max(v) for v in per_n.values()]) if per_n else float("nan"),
                        direct=np.mean(dg) if dg else float("nan"), grid_hit=np.mean(go) if go else float("nan"),
                        wins=sum(r["win"] for r in rs), t=np.mean([r["t_grow"] + r["t_finish"] for r in rs])))
    out.sort(key=lambda d: -d["gap"])
    return out


def show(tab, keys):
    print(" ".join(f"{k:>11}" for k in keys) + "  runs   gap  gap(best seed)  direct  grid-hit  wins   s/run")
    for d in tab:
        print(" ".join(f"{str(d[k]):>11}" for k in keys) +
              f" {d['runs']:5d} {d['gap']:6.3f} {d['gap_bestseed']:8.3f} {d['direct']:13.3f} {d['grid_hit']:7.2f}"
              f" {d['wins']:5d} {d['t']:7.1f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("results")
    p.add_argument("--best", required=True)
    p.add_argument("--by", default="")
    p.add_argument("--top", type=int, default=15)
    a = p.parse_args()
    best = {int(k): v for k, v in json.load(open(a.best)).items()}
    rows = load(a.results, best)
    present = [k for k in AXES if len({r.get(k) for r in rows}) > 1]
    for k in present:
        print(f"\n== by {k}")
        show(table(rows, [k]), [k])
    if a.by:
        keys = a.by.split(",")
        print(f"\n== by {a.by}")
        show(table(rows, keys), keys)
    print(f"\n== top {a.top} settings")
    show(table(rows, present)[:a.top], present)
    per_n = defaultdict(list)
    for r in rows:
        per_n[r["n"]].append(r["best"])
    print("\n== per n: best run vs grid vs best file")
    for n in sorted(per_n):
        print(f"n={n:4d} grow {min(per_n[n]):.5f} grid {math.ceil(math.sqrt(n))} best {best[n]:.5f}")
