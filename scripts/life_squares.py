"""Game-of-Life packing: inventors evolve local neighbour rules, then the winners sweep every n.

Examples:
    # Tournament: every inventor evolves a rule on the training n (from the stored packings),
    # then each rule is scored on held-out n it never saw. Rules go to the library.
    SQUARES_BEST_DIR=/mnt/project-files/squares/best python -m scripts.life_squares tournament \
        --train 26,37,50,65,82,101 --held 27,38,51,66,83,102 --gens 25

    # Same from loose random layouts (does the rule alone find good packings?)
    python -m scripts.life_squares tournament --mode random --train 20,30,40,55 --held 25,35,45,60

    # Play the library's best rule (or --rule NAME) on every stored n and offer improvements
    python -m scripts.life_squares sweep --n 5-1000 --seeds 0,1,2 --offer

    # Baseline for the same budget: the squeeze finish alone, no rule (rule = rep only, no melt)
    python -m scripts.life_squares sweep --n 5-200 --baseline
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np

from wm import best_store, life
from wm import life_evolve as le
from wm.squares import puzzle_score

LIB = le.LIBRARY


def parse_ns(spec: str) -> list[int]:
    ns = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        ns.extend(range(int(a), int(b or a) + 1))
    return ns


def held_out(c, ns, mode, steps, workers):
    res = le.score_rule(c, ns, mode, steps, workers=workers)
    ratios = {n: res[n][0] / res[n][1] for n in ns}
    return float(np.mean([min(r, 1.2) for r in ratios.values()])), ratios, res


def tournament(args):
    train, held = parse_ns(args.train), parse_ns(args.held)
    names = args.inventors.split(",") if args.inventors else list(le.INVENTORS)
    LIB.parent.mkdir(parents=True, exist_ok=True)
    table = []
    for name in names:
        inv = le.INVENTORS[name]
        t0 = time.time()
        c, fit, per_n = le.evolve(inv, train, args.mode, args.gens, args.popsize, steps=args.steps,
                                  seed=args.seed, workers=args.workers)
        h, ratios, res = held_out(c, held, args.mode, args.steps, args.workers)
        for n, (side, ref, pk) in res.items():
            if pk is not None and side < per_n.get(n, (math.inf,))[0]:
                per_n[n] = (side, *pk)
        won = le.offer_all(per_n, f"life:{name}") if args.offer else []
        le.save_rule(LIB, f"{name}-{args.mode}-s{args.seed}", inv, c, fit, train, args.mode,
                     held={"mean": h, "ratios": ratios})
        table.append((name, fit, h, len(won), time.time() - t0))
        print(f"== {name}: train {fit:.5f}  held-out {h:.5f}  wins {won}  ({time.time() - t0:.0f}s)", flush=True)
        print("   rule", json.dumps({k: round(v, 5) for k, v in life.as_dict(c).items()}), flush=True)
    print("\ninventor   train     held-out  wins")
    for name, fit, h, w, _ in sorted(table, key=lambda r: r[2]):
        print(f"{name:9s}  {fit:.5f}  {h:.5f}  {w}")


def pick_rule(name):
    lib = json.loads(LIB.read_text())
    if name:
        e = [r for r in lib if r["name"] == name][-1]
    else:
        e = min(lib, key=lambda r: r["held_out"]["mean"] if r.get("held_out") else r["fitness"])
    return e["name"], life.from_dict(e["rule"])


def sweep(args):
    ns = [n for n in parse_ns(args.n) if args.mode != "store" or best_store.load(n) is not None]
    if args.baseline:
        name, c = "baseline", life.from_dict({"melt": 0.0, "shrink": 1e-5, "rep": 0.5})
        steps = 1
    else:
        name, c = pick_rule(args.rule)
        steps = args.steps
    seeds = tuple(int(s) for s in args.seeds.split(","))
    t0 = time.time()
    res = le.score_rule(c, ns, args.mode, steps, seeds, workers=args.workers)
    per_n = {n: (side, *pk) if pk is not None else (side, None) for n, (side, ref, pk) in res.items()}
    better = {n: (res[n][1], res[n][0]) for n in ns if res[n][0] < res[n][1] - 1e-9}
    ratios = [res[n][0] / res[n][1] for n in ns]
    print(f"rule {name} on {len(ns)} n, seeds {seeds}: median ratio {np.median(ratios):.5f}, "
          f"{sum(r < 1 for r in ratios)} below store ({time.time() - t0:.0f}s)")
    for n, (ref, side) in sorted(better.items()):
        print(f"  n={n}: {ref:.6f} -> {side:.6f}  (score {puzzle_score(ref, n) - puzzle_score(side, n):+.5f})")
    if args.offer:
        won = le.offer_all({n: per_n[n] for n in better}, f"life:{name}")
        print("accepted:", won, "store total", round(best_store.total_score(), 5))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tournament")
    t.add_argument("--train", default="26,37,50,65,82,101")
    t.add_argument("--held", default="27,38,51,66,83,102")
    t.add_argument("--inventors", default="")
    t.add_argument("--gens", type=int, default=25)
    t.add_argument("--popsize", type=int, default=12)
    t.add_argument("--offer", action="store_true")
    s = sub.add_parser("sweep")
    s.add_argument("--n", default="5-1000")
    s.add_argument("--rule", default="")
    s.add_argument("--seeds", default="0")
    s.add_argument("--offer", action="store_true")
    s.add_argument("--baseline", action="store_true")
    for p in (t, s):
        p.add_argument("--mode", default="store", choices=["store", "drop", "random"])
        p.add_argument("--steps", type=int, default=600)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    tournament(args) if args.cmd == "tournament" else sweep(args)


if __name__ == "__main__":
    main()
