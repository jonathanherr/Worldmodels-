"""Build known-construction packings for every n up to 1000 and offer them to the best-per-n store.

1. For every candidate side s (of the form (i + j sqrt2)/2, fractional part >= 0.5, up to 32)
   try the tilted-block family over a pruned set of parameters and keep the largest count.
2. A construction of N squares in side s also gives, by wrapping L layers and dropping squares,
   every n <= N + (floor(s) + t)^2 - floor(s)^2 in side s + t. For each n take the smallest side.
3. Build each winner, polish it to a 1e-9 gap, check it with Kaggle's metric and offer it to
   wm.best_store (which keeps it only if it beats what is stored).

The sweep is cached in runs/constructions/sweep.json, so rebuilding the packings is quick.
"""

import argparse
import json
import math
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from wm import best_store
from wm.constructions import candidate_sides, configs, l_wrap, tilted_block
from wm.squares import MAX_N, polish


def sweep_one(s):
    best = (0, None)
    for cfg in configs(s):
        n = len(tilted_block(s, *cfg)[0])
        if n > best[0]:
            best = (n, cfg)
    return s, best[0], best[1]


def sweep(s_max, workers, cache: Path):
    done = {}
    if cache.exists():
        done = {round(r["s"], 12): r for r in json.loads(cache.read_text())}
    todo = [s for s in candidate_sides(s_max, 0.5) if round(s, 12) not in done]
    # Largest sides first: they take longest, so the pool stays busy to the end.
    with Pool(workers) as pool:
        for k, (s, n, cfg) in enumerate(pool.imap_unordered(sweep_one, todo[::-1], chunksize=1)):
            done[round(s, 12)] = {"s": s, "count": n, "cfg": cfg}
            if k % 20 == 0:
                cache.write_text(json.dumps(sorted(done.values(), key=lambda r: r["s"])))
                print(f"swept {len(done)} sides, last s={s:.4f} count={n}", flush=True)
    cache.write_text(json.dumps(sorted(done.values(), key=lambda r: r["s"])))
    return sorted(done.values(), key=lambda r: r["s"])


def plan(results, max_n=MAX_N):
    """{n: (side, base record, layers)} with the smallest side found for each n."""
    best = {}
    for r in results:
        s, count = r["s"], r["count"]
        k = math.floor(s + 1e-9)
        excess = count - k * k
        if excess <= 0:
            continue
        for t in range(0, 40):
            K = k + t
            for e in range(1, excess + 1):
                n = K * K + e
                if n > max_n or e >= 2 * K + 1:
                    continue
                if n not in best or s + t < best[n][0] - 1e-12:
                    best[n] = (s + t, r, t)
    return best


def build(n, rec, layers):
    s = rec["s"]
    p, q, shift, order = rec["cfg"]
    x, y, deg = tilted_block(s, p, q, tuple(shift), order)
    x, y, deg = l_wrap(x, y, deg, s, layers)
    # Drop surplus squares: keep the axis-aligned ones first (they sit on the outside).
    x, y, deg = x[:n], y[:n], deg[:n]
    return polish(x, y, deg, gap=1e-9)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--s-max", type=float, default=32.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cache", default="runs/constructions/sweep.json")
    ap.add_argument("--max-n", type=int, default=MAX_N)
    args = ap.parse_args()

    cache = Path(args.cache)
    cache.parent.mkdir(parents=True, exist_ok=True)
    results = sweep(args.s_max, args.workers, cache)
    chosen = plan(results, args.max_n)
    gained = 0.0
    for n in sorted(chosen):
        side, rec, layers = chosen[n]
        if side >= best_store.best_side(n) - 1e-9:
            continue
        x, y, deg, report = build(n, rec, layers)
        p, q, shift, order = rec["cfg"]
        name = f"construction:tilted-block(p={p},q={q},shift={tuple(shift)},{order})" + (f"+L{layers}" if layers else "")
        res = best_store.offer(n, x, y, deg, name)
        if res["accepted"]:
            gained += (res["previous"] - res["side"]) / np.sqrt(n)
        print(f"n={n:4d} side {res['side'] or float('nan'):.6f} (planned {side:.6f}) prev {res['previous']:.6f} {res['reason']}")
    print(f"score gained {gained:.4f}; store total now {best_store.total_score():.6f}")
