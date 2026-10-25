"""Sub-divided n: rebuild larger n from stored smaller packings (wm/morph.compose_2x2) until nothing improves.

    SQUARES_BEST_DIR=/mnt/project-files/squares/best python -m scripts.compose_squares --offer

Each round reads the store, makes s(n) monotone (a packing for m >= n with squares dropped is a
packing for n), lists every n some 2x2 composition beats, builds those packings and offers them.
Wins feed later rounds, so one improvement can cascade up through n.
"""

import argparse
from functools import lru_cache

import numpy as np

from wm import best_store, morph
from wm.squares import puzzle_score


def monotone_sides(max_n=1000):
    raw = np.array([0.0] + [best_store.best_side(n) for n in range(1, max_n + 1)])
    src = np.arange(max_n + 1)
    for n in range(max_n - 1, 0, -1):
        if raw[src[n + 1]] < raw[src[n]]:
            src[n] = src[n + 1]
    return raw[src], src


def make_packings(src):
    @lru_cache(maxsize=None)
    def packings(m):
        """Stored packing of src[m] >= m squares with the extras dropped, in [0, L]^2."""
        k = int(src[m])
        xyd = best_store.load_xyd(k)
        if xyd is None:
            g = int(np.ceil(np.sqrt(k)))
            i = np.arange(m)
            return (i % g + 0.5).astype(float), (i // g + 0.5).astype(float), np.zeros(m), float(g)
        x, y, d = xyd
        t = np.deg2rad(d)
        w = 0.5 * (np.abs(np.cos(t)) + np.abs(np.sin(t)))
        x0, y0 = (x - w).min(), (y - w).min()
        x, y = x - x0, y - y0
        L = max((x + w).max(), (y + w).max())
        return x[:m], y[:m], t[:m], float(L)
    return packings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-n", type=int, default=1000)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--offer", action="store_true")
    a = ap.parse_args()
    print(f"store total {best_store.total_score(max_n=a.max_n):.6f}")
    for rnd in range(a.rounds):
        sides, src = monotone_sides(a.max_n)
        packings = make_packings(src)
        wins = morph.composition_wins(sides, a.max_n)
        print(f"round {rnd}: {len(wins)} candidate n")
        if not wins or not a.offer:
            for n, (S, i, j) in sorted(wins.items()):
                print(f"  n={n} {S:.6f} < {sides[n]:.6f} from {i}+{j}")
            break
        accepted = 0
        for n, (S, i, j) in sorted(wins.items()):
            x, y, t, _ = morph.compose_2x2(i, j, sides, packings)
            assert len(x) == n, (n, len(x))
            r = best_store.offer(n, x, y, np.rad2deg(t), f"compose:{i}+{j}")
            gain = puzzle_score(sides[n], n) - puzzle_score(r["side"], n) if r["accepted"] else 0.0
            accepted += r["accepted"]
            print(f"  n={n} from {i}+{j}: side {r['side']} prev {sides[n]:.6f} {r['reason']} gain {gain:.5f}")
        if not accepted:
            break
    print(f"store total {best_store.total_score(max_n=a.max_n):.6f}")


if __name__ == "__main__":
    main()
