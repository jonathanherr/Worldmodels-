"""Compose rectangle packings into square packings and offer the winners to the best-per-n store.

Uses the S(w, L) table from scripts/rect_table.py (a lower bound on how many unit squares
wm.rectpack.rect_pack fits in a w x L rectangle) to score two families quickly:

  grid-L   side m + f: an integer grid (m - c2) x (m - c1) with a top strip (c1 + f) x s and a
           right strip (c2 + f) x (m - c1) (wm.rectpack.compose)
  wrap     any stored packing of side s' wrapped in an L of non-integer width w
           (wm.rectpack.wrap_core), side s' + w

Every candidate that beats the store on paper is then built for real, trimmed to n squares,
checked with Kaggle's metric and offered to the store.
"""

import argparse
import json
import math
from collections import defaultdict

import numpy as np

from wm import best_store
from wm.rectpack import Layout, compose, wrap_core

GRID_L = 0.25


def load_table(path):
    S = defaultdict(dict)
    for r in json.loads(open(path).read()):
        S[round(r["w"], 4)][round(r["L"], 2)] = r["n"]
    return S


def make_lookup(S):
    def s_lb(w, L):
        w = round(w, 4)
        Lg = round(math.floor(L / GRID_L + 1e-9) * GRID_L, 2)
        row = S.get(w)
        axis = int(math.floor(w + 1e-12)) * int(math.floor(L + 1e-12))
        if row is None or Lg not in row:
            return axis
        return max(row[Lg], axis)
    return s_lb


def candidates(S, max_n):
    s_lb = make_lookup(S)
    ws = sorted(S)
    fs = sorted({round(w - math.floor(w), 4) for w in ws})
    best = {}  # n -> (side, recipe)

    def offer(count, side, recipe):
        for n in range(1, min(count, max_n) + 1):
            if side < best.get(n, (math.ceil(math.sqrt(n)) - 1e-12,))[0]:
                best[n] = (side, recipe)

    for m in range(2, 33):
        for f in fs:
            s = m + f
            for c1 in range(0, m + 1):
                for c2 in range(0, m - c1 + 1):
                    cnt = (m - c2) * (m - c1)
                    if c1:
                        cnt += s_lb(c1 + f, s)
                    if c2:
                        cnt += s_lb(c2 + f, m - c1)
                    offer(cnt, s, ("grid-L", s, c1, c2))
    cores = [(k * k, float(k), None) for k in range(1, 33)]
    for n in range(1, max_n + 1):
        e = best_store.load(n)
        if e:
            cores.append((n, e["side"], n))
    for n0, s0, src in cores:
        for w in ws:
            s = s0 + w
            if s > 32.5:
                continue
            offer(n0 + s_lb(w, s0) + s_lb(w, s), s, ("wrap", s0, w, src, n0))
    return best


def build(recipe):
    if recipe[0] == "grid-L":
        _, s, c1, c2 = recipe
        x, y, d = compose(Layout(s, c1, c2))
        return x, y, d, s
    _, s0, w, src, n0 = recipe
    if src is None:
        k = int(round(s0))
        g = np.arange(k) + 0.5
        gx, gy = np.meshgrid(g, g, indexing="ij")
        core = (gx.ravel(), gy.ravel(), np.zeros(k * k))
    else:
        core = best_store.load_xyd(src)
    x, y, d = wrap_core(core, s0, w)
    return x, y, d, s0 + w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="runs/rectpack/table.json")
    ap.add_argument("--max-n", type=int, default=1000)
    ap.add_argument("--dry", action="store_true", help="only list paper wins")
    a = ap.parse_args()
    S = load_table(a.table)
    best = candidates(S, a.max_n)
    wins = []
    for n, (side, recipe) in sorted(best.items()):
        cur = best_store.best_side(n)
        if side < cur - 1e-9:
            wins.append((n, side, cur, recipe))
    gain = sum((cur - side) / math.sqrt(n) for n, side, cur, _ in wins)
    print(f"{len(wins)} paper wins, score gain on paper {gain:.4f}")
    for w in wins[:40]:
        print(w)
    if a.dry:
        return
    built = {}
    accepted = 0
    for n, side, cur, recipe in wins:
        key = repr(recipe)
        if key not in built:
            built[key] = build(recipe)
        x, y, d, s = built[key]
        if len(x) < n:
            print("short", n, len(x), recipe)
            continue
        # keep the n squares nearest the bottom-left corner; dropping squares never hurts
        order = np.argsort(np.maximum(x, y), kind="stable")[:n]
        r = best_store.offer(n, x[order], y[order], d[order], source=f"rectpack:{recipe[0]}")
        accepted += r["accepted"]
        print(n, round(side, 5), "->", r)
    print("accepted", accepted, "store total now", round(best_store.total_score(), 4))


if __name__ == "__main__":
    main()
