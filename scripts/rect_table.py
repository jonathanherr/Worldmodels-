"""Tabulate S(w, L): the most unit squares rect_pack fits in a w x L rectangle.

Widths w = c + f for c = 1..8 and f in [0.70, 0.99]; lengths on a 0.25 grid. Because S grows with
L, S(w, floor_to_grid(L)) is a valid lower bound for any L, which is all the composer needs.
"""

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from wm.rectpack import rect_pack


def one(args):
    w, L = args
    x, _, _, name = rect_pack(w, L)
    return w, L, len(x), name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/rectpack/table.json")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--fstep", type=float, default=0.02)
    ap.add_argument("--lmax", type=float, default=33.0)
    a = ap.parse_args()
    ws = sorted({round(c + f, 4) for c in range(1, 9) for f in np.arange(0.70, 0.995, a.fstep)})
    Ls = [round(L, 2) for L in np.arange(3.0, a.lmax + 1e-9, 0.25)]
    jobs = [(w, L) for w in ws for L in Ls]
    jobs.sort(key=lambda j: -j[0] * j[1])  # big ones first keeps the pool busy
    with Pool(a.workers) as p:
        rows = p.map(one, jobs, chunksize=4)
    Path(a.out).write_text(json.dumps([{"w": w, "L": L, "n": n, "how": h} for w, L, n, h in rows]))
    print(len(rows), "entries ->", a.out)


if __name__ == "__main__":
    main()
