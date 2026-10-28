"""Clustered-block compositions (wm/blockpack.py) against a submission file or the store.

    python -m scripts.block_squares --csv submission.csv            # find n a composition beats
    python -m scripts.block_squares --csv submission.csv --cap 100  # only blocks of <= 100 squares
    SQUARES_BEST_DIR=... python -m scripts.block_squares --offer    # read and offer to the store

--cap answers "how much of the score do small blocks explain": records above the cap are
replaced by the grid before composing, and the composed total is printed.
"""

import argparse

import numpy as np

from wm import best_store
from wm.blockpack import Blocks, BlockPacker
from wm.squares import puzzle_score


def load_csv(path, max_n=1000):
    import pandas as pd

    df = pd.read_csv(path, dtype=str)
    packs = {}
    for n, g in df.groupby(df.n.astype(int)):
        x, y, t = (g[c].str.lstrip("s").astype(float).values for c in ("x", "y", "angle"))
        w = 0.5 * (np.abs(np.cos(t)) + np.abs(np.sin(t)))
        x0, y0 = (x - w).min(), (y - w).min()
        packs[n] = (x - x0, y - y0, t, float(max((x + w).max() - x0, (y + w).max() - y0)))
    return packs


def load_store(max_n=1000):
    packs = {}
    for n in range(1, max_n + 1):
        xyd = best_store.load_xyd(n)
        if xyd is None:
            g = int(np.ceil(np.sqrt(n)))
            i = np.arange(n)
            packs[n] = ((i % g + 0.5).astype(float), (i // g + 0.5).astype(float), np.zeros(n), float(g))
            continue
        x, y, d = xyd
        t = np.deg2rad(d)
        w = 0.5 * (np.abs(np.cos(t)) + np.abs(np.sin(t)))
        x0, y0 = (x - w).min(), (y - w).min()
        packs[n] = (x - x0, y - y0, t, float(max((x + w).max() - x0, (y + w).max() - y0)))
    return packs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="read packings from a submission CSV instead of the store")
    ap.add_argument("--max-n", type=int, default=1000)
    ap.add_argument("--cap", type=int, default=None, help="use only blocks of <= cap squares")
    ap.add_argument("--offer", action="store_true")
    a = ap.parse_args()
    packs = load_csv(a.csv, a.max_n) if a.csv else load_store(a.max_n)
    raw = np.array([0.0] + [packs[n][3] for n in range(1, a.max_n + 1)])
    grid = np.ceil(np.sqrt(np.arange(a.max_n + 1)))
    lib = raw.copy()
    if a.cap is not None:
        lib[a.cap + 1:] = grid[a.cap + 1:]
    blocks = Blocks(lib, lambda k: packs[k])
    bp = BlockPacker(blocks)
    total, wins = 0.0, []
    for n in range(1, a.max_n + 1):
        side = bp.min_side(n, grid[n] + 1e-6, tol=1e-7) or grid[n]
        if a.cap is not None and n <= a.cap:
            side = min(side, raw[n])
        total += puzzle_score(side, n)
        if side < raw[n] - 1e-6:
            wins.append((n, side))
    base = sum(puzzle_score(raw[n], n) for n in range(1, a.max_n + 1))
    print(f"input total {base:.6f}, composed total {total:.6f}, {len(wins)} n beat the input")
    for n, side in wins:
        line = f"  n={n} composed {side:.6f} < {raw[n]:.6f}"
        if a.offer:
            x, y, t = bp.build(side + 1e-7, n)
            r = best_store.offer(n, x, y, np.rad2deg(t), "blocks:shelf")
            line += f" offer {r['reason']} side {r['side']}"
        print(line)


if __name__ == "__main__":
    main()
