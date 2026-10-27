"""Soft-shape melt-and-harden trials on a base submission; report any packing that beats it.

    python -m scripts.soft_squares BASE.csv --ns 147 335 --minutes 20 --out wins.json

Each trial picks a random setting (shrink of the box, shape stiffness schedule, noise, whether
squares may stretch into rectangles), runs `wm.soft.soft_trial` and compares the rigid result's
side with the base. Wins are re-checked with the official metric before they are written.
"""

import argparse
import json
import os
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from multiprocessing import Pool

import numpy as np
import pandas as pd

from wm import soft
from wm.fault_dp import check_rows, normalize, side_of, to_rows

PACKS = {}

SHRINKS = (2e-5, 1e-4, 5e-4, 2e-3, 1e-2)
SCHEDULES = {
    "gentle": (30.0, 100.0, 1e3, 1e4),
    "melt": (1.0, 10.0, 100.0, 1e3, 1e4),
    "deep": (0.1, 1.0, 10.0, 100.0, 1e3, 1e4),
}
TEMPS = (0.0, 1e-3, 5e-3, 2e-2)
SHRINKS_USED = list(SHRINKS)


def trial(args):
    n, seed = args
    rng = np.random.default_rng(seed)
    p = PACKS[n]
    L0 = side_of(p)
    shrink = float(rng.choice(SHRINKS_USED))
    sched = str(rng.choice(list(SCHEDULES)))
    temp = float(rng.choice(TEMPS))
    square = bool(rng.integers(2))
    steps = int(rng.choice((100, 300)))
    t0 = time.time()
    r, deficit = soft.soft_trial(p, L0, shrink, SCHEDULES[sched], steps, temp, square, seed)
    out = {"n": n, "seed": seed, "shrink": shrink, "sched": sched, "temp": temp, "square": square,
           "steps": steps, "base": L0, "side": None, "secs": round(time.time() - t0, 1),
           "max_deficit": float(deficit.max()), "n_shrunk": int((deficit > 1e-3).sum())}
    if r is not None:
        x, y, t, _ = r
        q = normalize(np.c_[x, y, t])
        out["side"] = side_of(q)
        if out["side"] < L0 - 1e-9:
            out["p"] = q.tolist()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("--ns", type=int, nargs="+", required=True)
    ap.add_argument("--minutes", type=float, default=10)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="soft_wins.json")
    ap.add_argument("--log", default="soft_log.jsonl")
    ap.add_argument("--shrinks", type=float, nargs="+", default=list(SHRINKS))
    a = ap.parse_args()

    df = pd.read_csv(a.base)
    packs = {int(n): normalize(g.sort_values("square_id")[["x", "y", "angle"]].to_numpy(float))
             for n, g in df[df.n.isin(a.ns)].groupby("n")}
    PACKS.update(packs)
    SHRINKS_USED[:] = a.shrinks
    wins = {}
    t_end = time.time() + 60 * a.minutes
    jobs = ((n, a.seed + 1000 * k + n) for k in range(10 ** 6) for n in a.ns)
    with Pool(a.workers) as pool, open(a.log, "a") as log:
        for res in pool.imap_unordered(trial, jobs):
            log.write(json.dumps({k: v for k, v in res.items() if k != "p"}) + "\n")
            log.flush()
            if "p" in res:
                n = res["n"]
                side, why = check_rows(n, to_rows(n, np.array(res["p"])))
                print(f"candidate n={n} side={res['side']:.9f} base={res['base']:.9f} official={side} {why}",
                      flush=True)
                if side and side < res["base"] and (n not in wins or side < wins[n]["side"]):
                    wins[n] = {"side": side, "base": res["base"], "p": res["p"]}
                    json.dump(wins, open(a.out, "w"))
            if time.time() > t_end:
                pool.terminate()
                break
    print("wins:", {n: (w["base"] - w["side"]) for n, w in wins.items()})


if __name__ == "__main__":
    main()
