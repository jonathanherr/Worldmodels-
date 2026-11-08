"""Sweep drunk-walk grid starts over n in parallel and compare with a base submission.

    OMP_NUM_THREADS=1 python -m scripts.drunkwalk_squares --n 10-120 --seeds 8 \
        --base submission.csv --out drunk_results.json

Each worker runs every (shape, sigma, tilt, drop, seed) trial for one n and keeps the smallest
side. A side below the base file's side is checked with the official metric before it counts
as a win; winning packings are written to --out with their exact CSV strings.
"""

import argparse
import itertools
import json
import os
import time
from multiprocessing import Pool

import numpy as np

from wm.drunkwalk import shapes, trial
from wm.squares import official_check


def parse_ns(spec):
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def base_sides(path):
    import pandas as pd

    d = pd.read_csv(path)
    sides = {}
    for n, g in d.groupby("n"):
        w = 0.5 * (np.abs(np.cos(g.angle.values)) + np.abs(np.sin(g.angle.values)))
        x, y = g.x.values, g.y.values
        sides[int(n)] = float(max((x + w).max() - (x - w).min(), (y + w).max() - (y - w).min()))
    return sides


def run_n(job):
    n, seeds, sigmas, tilts, drops, base = job
    t0 = time.time()
    best, sides = None, []
    for seed, (r, c), sig, tilt, drop in itertools.product(range(seeds), shapes(n), sigmas, tilts, drops):
        res = trial(n, r, c, sig, seed * 1000 + n, tilt=tilt, drop=drop, smooth=2)
        if res is None:
            continue
        sides.append(res[0])
        if best is None or res[0] < best[0]:
            best = (res[0], res[1:], dict(r=r, c=c, sigma=sig, tilt=tilt, drop=drop, seed=seed))
    out = {"n": n, "grid": float(np.ceil(np.sqrt(n))), "base": base, "trials": len(sides),
           "below_grid": int(np.sum(np.array(sides) < np.ceil(np.sqrt(n)) - 1e-6)),
           "best": best[0] if best else None, "params": best[2] if best else None,
           "secs": round(time.time() - t0, 1), "win": False}
    if best and best[0] < base - 1e-9:
        x, y, t = best[1]
        rows = [{"id": k, "n": n, "square_id": k, "x": repr(float(x[k])), "y": repr(float(y[k])),
                 "angle": repr(float(t[k]))} for k in range(n)]
        check = official_check(rows)
        if check["valid"] and check["side"] < base - 1e-9:
            out.update(win=True, official=check["side"], rows=rows)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", required=True, help="e.g. 10-120,556")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--sigmas", default="0.3,0.6,1.0")
    ap.add_argument("--tilts", default="1,2.5")
    ap.add_argument("--drops", default="small,random")
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()
    base = base_sides(a.base)
    jobs = [(n, a.seeds, [float(s) for s in a.sigmas.split(",")], [float(s) for s in a.tilts.split(",")],
             a.drops.split(","), base[n]) for n in parse_ns(a.n)]
    jobs.sort(key=lambda j: -j[0])  # big n first so the pool finishes evenly
    results = []
    with Pool(a.workers) as pool:
        for res in pool.imap_unordered(run_n, jobs):
            results.append(res)
            flag = "WIN" if res["win"] else ""
            print(f"n={res['n']} best={res['best']:.5f} base={res['base']:.5f} grid={res['grid']:.0f} "
                  f"below_grid={res['below_grid']}/{res['trials']} {res['secs']}s {flag}", flush=True)
            json.dump(sorted(results, key=lambda r: r["n"]), open(a.out, "w"))


if __name__ == "__main__":
    main()
