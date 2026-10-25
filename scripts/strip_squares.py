"""Sweep the any-angle tilted-strip family (wm/strips.py) and offer packings that beat the store.

    SQUARES_BEST_DIR=/mnt/project-files/squares/best python -m scripts.strip_squares --offer

For every container side S = K + delta (delta on a grid) and every split of the container into an
A x B grid block plus a top and a right band, find the best band fills and record, for each n, the
smallest S that holds n squares. Splits tried: A = B = K - r (an L of two bands around a square
grid block), A and B one apart, and A = K, B = K - r (one top band over a full-width grid).
"""

import argparse
from multiprocessing import Pool

import numpy as np

from wm import best_store, strips
from wm.squares import puzzle_score

THETAS = np.deg2rad(np.linspace(0.1, 45, 450))


def sweep_side(S):
    K = int(np.floor(S))
    out = {}
    for r in range(0, K + 1):
        for A, B in ((K - r, K - r), (K, K - r), (K - r, K - r - 1), (K - r - 1, K - r)):
            if min(A, B) < 0:
                continue
            n, top, right = strips.layout_count(S, A, B, THETAS, phases=12)
            if n > K * K and (n not in out or out[n][0] > S):
                out[n] = (S, A, B, top, right)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k-min", type=int, default=4)
    ap.add_argument("--k-max", type=int, default=31)
    ap.add_argument("--step", type=float, default=0.01)
    ap.add_argument("--delta-min", type=float, default=0.3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--offer", action="store_true")
    a = ap.parse_args()
    sides = [K + d for K in range(a.k_min, a.k_max + 1)
             for d in np.arange(a.delta_min, 1.0, a.step)]
    best = {}
    with Pool(a.workers) as pool:
        for out in pool.imap_unordered(sweep_side, sides, chunksize=4):
            for n, v in out.items():
                if n <= 1000 and (n not in best or v[0] < best[n][0]):
                    best[n] = v
    # A packing for n also packs every smaller n.
    for n in sorted(best, reverse=True):
        if n - 1 > 0 and (n - 1 not in best or best[n - 1][0] > best[n][0]) and int(best[n][0]) ** 2 < n - 1:
            best[n - 1] = best[n]
    wins = 0
    for n in sorted(best):
        S, A, B, top, right = best[n]
        prev = best_store.best_side(n)
        if S >= prev - 1e-9:
            continue
        x, y, d = strips.layout(S, A, B, top, right)
        x, y, d = x[:n], y[:n], d[:n]
        msg = f"n={n} S={S:.4f} A={A} B={B} top={np.rad2deg(top[0]):.2f}deg right={np.rad2deg(right[0]):.2f}deg store {prev:.4f}"
        if a.offer:
            r = best_store.offer(n, x, y, d, f"strip:A{A}B{B}")
            gain = puzzle_score(prev, n) - puzzle_score(r["side"], n) if r["accepted"] else 0.0
            msg += f" -> {r['reason']} side {r['side']} gain {gain:.5f}"
            wins += r["accepted"]
        print(msg, flush=True)
    print(f"accepted {wins}")


if __name__ == "__main__":
    main()
