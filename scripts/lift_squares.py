"""Dimensional-relaxation solver (wm/lift.py): A/B against the flat optimizer, offer wins to the store.

Two experiments, each run with the lift and with its flat control on the same seeds:

    # Random starts: lift_compact vs compact, both finished by squeeze
    python -m scripts.lift_squares random --n 11,17,29,37,41 --seeds 16

    # Stored packings: ask for a box `shrink` smaller, lifted vs flat-with-jitter
    SQUARES_BEST_DIR=/mnt/project-files/squares/best python -m scripts.lift_squares store --n 26-60 --tries 24

Add --offer to send certified improvements to the store (SQUARES_BEST_DIR or solutions/).
Results are appended as JSON lines to --out.
"""

import argparse
import json
import multiprocessing as mp
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402

from scripts.evolve_squares import parse_ns  # noqa: E402
from wm import best_store  # noqa: E402
from wm.lift import lift_compact, lift_search  # noqa: E402
from wm.pack_opt import compact, squeeze  # noqa: E402
from wm.squares import polish  # noqa: E402


def _offer(n, x, y, t, source):
    px, py, pd, rep = polish(x, y, np.rad2deg(t), gap=1e-9)
    if not rep["valid"]:
        return {"accepted": False, "reason": "polish"}
    return best_store.offer(n, px, py, pd, source)


def run_random(args):
    n, seed, lift, offer = args
    rng = np.random.default_rng(seed)
    L0 = np.sqrt(n) * 1.2
    x, y, t = rng.uniform(0, L0, n), rng.uniform(0, L0, n), rng.uniform(-np.pi / 4, np.pi / 4, n)
    t0 = time.time()
    if lift:
        x, y, t, L = lift_compact(x, y, t, L0, z=rng.uniform(-0.5, 0.5, n))
    else:
        x, y, t, L = compact(x, y, t, L0)
    r = squeeze(x, y, t, L * 1.0005, rounds=25, step=0.001, method="lbfgs", maxiter=1500)
    side = float(r[3]) if r is not None else float("nan")
    out = {"exp": "random", "n": n, "seed": seed, "lift": lift, "side": side, "secs": time.time() - t0}
    if offer and r is not None and side < best_store.best_side(n) - 1e-9:
        out["offer"] = _offer(n, *r[:3], "lift:random" if lift else "flat:random")
    return out


def run_store(args):
    n, seed, lift, tries, shrink, offer = args
    xyd = best_store.load_xyd(n)
    if xyd is None:
        return {"exp": "store", "n": n, "seed": seed, "lift": lift, "skipped": "nothing stored"}
    x, y, d = xyd
    L = best_store.best_side(n)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    kw = dict(H0=1.0, lens=True) if lift else dict(H0=0.0, jitter=0.03)
    bx, by, bt, bL = lift_search(x, y, np.deg2rad(d), L, tries=tries, shrink=shrink, rng=rng, **kw)
    out = {"exp": "store", "n": n, "seed": seed, "lift": lift, "stored": L, "side": float(bL),
           "secs": time.time() - t0}
    if offer and bL < L - 1e-9:
        out["offer"] = _offer(n, bx, by, bt, "lift:store" if lift else "flat:store")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", choices=["random", "store"])
    ap.add_argument("--n", required=True)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--tries", type=int, default=24)
    ap.add_argument("--shrink", type=float, default=0.002)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--no-control", action="store_true", help="skip the flat control runs")
    ap.add_argument("--offer", action="store_true")
    ap.add_argument("--out", default="lift_results.jsonl")
    a = ap.parse_args()
    modes = [True] if a.no_control else [True, False]
    ns = parse_ns(a.n)
    if a.exp == "random":
        fn, jobs = run_random, [(n, s, m, a.offer) for n in ns for s in range(a.seeds) for m in modes]
    else:
        fn, jobs = run_store, [(n, s, m, a.tries, a.shrink, a.offer) for n in ns for s in range(a.seeds) for m in modes]
    with mp.get_context("spawn").Pool(a.workers) as pool, open(a.out, "a") as f:
        for r in pool.imap_unordered(fn, jobs):
            f.write(json.dumps(r) + "\n")
            f.flush()
            print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
