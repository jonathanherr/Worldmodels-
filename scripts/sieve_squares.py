"""A/B sweep of sieve-and-settle packings against plain shaking, compared with the best store.

    python -m scripts.sieve_squares --n 20-60 --seeds 16 --modes sieve,cells,shake --workers 4
    SQUARES_BEST_DIR=/mnt/project-files/squares/best python -m scripts.sieve_squares --n 29 --offer

Every run of every mode gets the same integrator budget and the same random container width W
(sampled just under the stored side, so the lid presses the squares into a box that must beat
the store). Results go to a JSON lines file; --offer sends certified improvements to the store.
"""

import os

# One BLAS thread per worker: with several workers, threaded BLAS inside scipy's L-BFGS
# oversubscribes the cores and runs ~100x slower.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402

from wm import best_store  # noqa: E402
from wm.sieve import close_box, settle  # noqa: E402


def parse_ns(spec):
    ns = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            ns.extend(range(int(a), int(b) + 1))
        else:
            ns.append(int(part))
    return ns


def one(job):
    n, mode, seed, args = job
    rng = np.random.default_rng([n, seed])
    best = best_store.best_side(n)
    # W from just above the store down to the calibration-model side (where we think the
    # leader is); the same W for every mode at this (n, seed).
    k = math.ceil(math.sqrt(n))
    target = min(k, math.sqrt(n + 0.97 * k))
    w_rng = np.random.default_rng([n, seed, 7])
    W = float(w_rng.uniform(min(target, best) - 0.02, best + 0.01))
    hole = float(w_rng.uniform(1.1, 1.4))
    temp = float(w_rng.choice([0.1, 0.3, 0.6]))
    t0 = time.time()
    init = None
    if mode == "seeded":
        xyd = best_store.load_xyd(n)
        if xyd is None:
            k = math.ceil(math.sqrt(n))
            xyd = (np.arange(n) % k + 0.5, np.arange(n) // k + 0.5, np.zeros(n))
        init = (xyd[0], xyd[1], np.deg2rad(xyd[2]))
    gravity = (0.0, -1.0)
    if mode.endswith("-corner"):  # e.g. sieve-corner: same scenario, container tilted 45 degrees
        mode, gravity = mode[:-7], (-np.sqrt(0.5), -np.sqrt(0.5))
    x, y, t = settle(n, W, rng, mode=mode, hole=hole, budget=args.budget, temp_shake=temp, init=init,
                     gravity=gravity)
    res = close_box(x, y, t)
    out = {"n": n, "mode": job[1], "seed": seed, "W": W, "hole": hole, "temp": temp,
           "best": best, "secs": round(time.time() - t0, 2)}
    if res is None:
        out["L"] = None
        return out, None
    x, y, t, L = res
    out["L"] = L
    tilt = np.abs(((t + np.pi / 4) % (np.pi / 2)) - np.pi / 4)
    out["tilted"] = int((tilt > 0.02).sum())
    return out, (x, y, np.rad2deg(t)) if L < best - 1e-9 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", default="20-40")
    ap.add_argument("--modes", default="sieve,cells,shake")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--budget", type=int, default=30000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="sieve_results.jsonl")
    ap.add_argument("--offer", action="store_true")
    args = ap.parse_args()
    jobs = [(n, m, s, args) for n in parse_ns(args.n) for s in range(args.seeds)
            for m in args.modes.split(",")]
    with Pool(args.workers) as pool, open(args.out, "a") as f:
        for out, pack in pool.imap_unordered(one, jobs):
            if pack is not None and args.offer:
                r = best_store.offer(out["n"], *pack, source=f"sieve:{out['mode']}")
                out["offer"] = r["reason"]
                if r["accepted"]:
                    print(f"WIN n={out['n']} {out['mode']} {r['previous']:.5f} -> {r['side']:.5f}", flush=True)
            f.write(json.dumps(out) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
