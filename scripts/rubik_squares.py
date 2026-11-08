"""Rubik's-cube slice/block turns + settle on stored packings; report any packing that beats the base.

    python -m scripts.rubik_squares BASE.csv --ns 147 335 --minutes 30 --out wins.json

Each trial scrambles a packing for n with 1..depth random cube moves (faces = its 8 dihedral
images plus the n-1 and n+1 packings), settles it with compact_squeeze, and compares the
result with the base side for however many squares it now holds. Wins are re-checked with the
official metric before they are written.
"""

import argparse
import json
import os
import time

# One BLAS/numba thread per worker: oversubscribed threads made trials ~25x slower.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from multiprocessing import Pool

import numpy as np
import pandas as pd

from wm import rubik
from wm.fault_dp import check_rows, to_rows

PACKS = {}
SIDES = {}


def _init(packs):
    PACKS.update(packs)
    SIDES.update({n: rubik.side_of(p) for n, p in packs.items()})


def trial(args):
    n, seed, depth, kinds = args
    rng = np.random.default_rng(seed)
    p = PACKS[n]
    L = SIDES[n]
    faces = [rubik.dihedral(p, L, k) for k in range(1, 8)]
    faces += [PACKS[m] for m in (n - 1, n + 1) if m in PACKS and abs(SIDES[m] - L) < 0.5]
    q, desc = p, []
    for _ in range(int(rng.integers(1, depth + 1))):
        q2, d = rubik.random_move(q, L, rng, faces, kinds)
        if q2 is None:
            continue
        q, desc = q2, desc + [d]
    m = len(q)
    if m not in SIDES or not desc:
        return None
    r = rubik.settle(q, L)
    if r is None:
        return None
    qs, s = r
    return {"n": n, "m": m, "side": s, "base": SIDES[m], "moves": desc, "seed": seed,
            "p": qs.tolist() if s < SIDES[m] - 1e-9 else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("--ns", type=int, nargs="+", required=True)
    ap.add_argument("--minutes", type=float, default=10)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--kinds", nargs="+", default=["slice", "block"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="rubik_wins.json")
    a = ap.parse_args()

    df = pd.read_csv(a.base)
    need = set()
    for n in a.ns:
        need |= {n - 1, n, n + 1}
    packs = {}
    for n, g in df[df.n.isin(need)].groupby("n"):
        packs[int(n)] = rubik.normalize(g.sort_values("square_id")[["x", "y", "angle"]].to_numpy(float))
    closest, wins, done = {}, {}, 0
    t_end = time.time() + 60 * a.minutes

    def jobs():
        k = 0
        while time.time() < t_end:  # small batches: imap would drain an endless generator up front
            batch = [(a.ns[(k + i) % len(a.ns)], a.seed * 10_000_000 + k + i, a.depth, tuple(a.kinds))
                     for i in range(2 * a.workers)]
            k += len(batch)
            yield from pool.imap_unordered(trial, batch)

    with Pool(a.workers, initializer=_init, initargs=(packs,)) as pool:
        for r in jobs():
            done += 1
            if r is None:
                continue
            gap = r["side"] - r["base"]
            key = r["m"]
            if key not in closest or gap < closest[key][0]:
                closest[key] = (gap, r["n"], r["moves"])
            if r["p"] is not None:
                p = np.array(r["p"])
                side, reason = check_rows(r["m"], to_rows(r["m"], p))
                print(f"candidate n={r['m']} from {r['n']} side {r['side']:.9f} base {r['base']:.9f} "
                      f"official {side} {reason or ''} {r['moves']}", flush=True)
                if side is not None and side < r["base"] and (key not in wins or side < wins[key]["side"]):
                    wins[key] = {"side": side, "base": r["base"], "from": r["n"], "moves": r["moves"], "p": r["p"]}
                    json.dump(wins, open(a.out, "w"))
    print(f"trials {done}")
    for m in sorted(closest):
        g, n, mv = closest[m]
        print(f"n={m}: closest gap {g:+.2e} (from {n}) {mv}")
    print(f"wins: {sorted(wins)}")


if __name__ == "__main__":
    main()
