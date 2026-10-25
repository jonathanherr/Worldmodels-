"""Neighbouring-n morphs (wm/morph.py): A/B fractional vs abrupt, offer wins to the store.

    # n from n+1 (remove a square) and from n-1 (insert one), 8 candidates each
    SQUARES_BEST_DIR=/mnt/project-files/squares/best python -m scripts.morph_squares --n 10-60 --moves remove,insert

Each job is one (n, move, candidate, fractional?) and appends a JSON line to --out. Add --offer
to send certified improvements to the store (SQUARES_BEST_DIR or solutions/).
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
from wm import best_store, morph  # noqa: E402
from wm.squares import polish  # noqa: E402


def packing(n):
    """Stored packing for n as (x, y, t radians, L) in [0, L]^2, or the grid if nothing is stored."""
    xyd = best_store.load_xyd(n)
    if xyd is None:
        k = int(np.ceil(np.sqrt(n)))
        i = np.arange(n)
        return (i % k + 0.5).astype(float), (i // k + 0.5).astype(float), np.zeros(n), float(k)
    x, y, d = xyd
    t = np.deg2rad(d)
    w = 0.5 * (np.abs(np.cos(t)) + np.abs(np.sin(t)))
    x0, y0 = (x - w).min(), (y - w).min()
    x, y = x - x0, y - y0
    L = max((x + w).max(), (y + w).max())
    return x, y, t, float(L)


def jobs_for(n, moves, k, seed):
    rng = np.random.default_rng(seed + n)
    out = []
    if "remove" in moves and n + 1 <= 1000:
        x, y, t, L = packing(n + 1)
        for r in morph.removal_candidates(x, y, t, L, k=k, rng=rng):
            for frac in (False, True):
                out.append((n, "remove", r, frac))
    if "insert" in moves and n - 1 >= 1:
        x, y, t, L = packing(n - 1)
        for h, hole in enumerate(morph.holes(x, y, t, L, k=max(1, k // 2))):
            for ang in (0.0, np.pi / 4):
                for frac in (False, True):
                    out.append((n, "insert", (h, hole, ang), frac))
    return out


def run(job):
    (n, move, cand, frac), offer = job
    t0 = time.time()
    if move == "remove":
        x, y, t, L = packing(n + 1)
        r = morph.remove_fractional(x, y, t, L, cand) if frac else morph.remove_abrupt(x, y, t, L, cand)
        tag = cand
    else:
        x, y, t, L = packing(n - 1)
        h, hole, ang = cand
        f = morph.insert_fractional if frac else morph.insert_abrupt
        r = f(x, y, t, L, hole, angle=ang)
        tag = f"h{h}a{int(round(np.rad2deg(ang)))}"
    best = best_store.best_side(n)
    side = float(r[3]) if r is not None else float("nan")
    out = {"n": n, "move": move, "cand": tag, "frac": frac, "side": side, "best": best,
           "secs": round(time.time() - t0, 1)}
    if offer and r is not None and side < best - 1e-9:
        px, py, pd, rep = polish(r[0], r[1], np.rad2deg(r[2]), gap=1e-9)
        src = f"morph:{move}:{'frac' if frac else 'abrupt'}"
        out["offer"] = best_store.offer(n, px, py, pd, src) if rep["valid"] else {"accepted": False, "reason": "polish"}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", default="10-40")
    ap.add_argument("--moves", default="remove,insert")
    ap.add_argument("--k", type=int, default=8, help="candidates per move")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--offer", action="store_true")
    ap.add_argument("--out", default="morph_results.jsonl")
    a = ap.parse_args()
    moves = a.moves.split(",")
    jobs = [(j, a.offer) for n in parse_ns(a.n) for j in jobs_for(n, moves, a.k, a.seed)]
    print(f"{len(jobs)} jobs")
    with mp.get_context("spawn").Pool(a.workers) as pool, open(a.out, "a") as fh:
        for res in pool.imap_unordered(run, jobs):
            fh.write(json.dumps(res) + "\n")
            fh.flush()
            gap = res["side"] - res["best"]
            mark = " WIN" if gap < -1e-9 else ""
            print(f"n={res['n']} {res['move']:6s} {str(res['cand']):6s} frac={int(res['frac'])} "
                  f"side={res['side']:.5f} best={res['best']:.5f} gap={gap:+.5f} {res['secs']}s{mark}"
                  + (f" offer={res['offer']['reason']}" if "offer" in res else ""), flush=True)


if __name__ == "__main__":
    main()
