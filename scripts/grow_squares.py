"""Sweep the asymmetric growth packer (wm/grow.py) over simultaneity, adjacency, direction and rate.

Each run seeds n tiny squares, grows them until they jam, then reports three sides:
  direct   1 / min(s) of the jammed state (pure growth, no polish),
  squeeze  that state squeezed with the repo's FIRE squeeze (keeps the jammed layout),
  compact  compact_squeeze from the jammed state (penalty compaction, then squeeze).
Every result is compared with the grid side ceil(sqrt n) and the best file's side for that n.

  python scripts/grow_squares.py --grid main --ns 10,15,20,27,30,37,40,50 --seeds 0,1,2 \
      --best best_sides.json --out grow_main.jsonl --workers 4
"""

import argparse
import itertools
import json
import math
import os
import time
from multiprocessing import Pool

import numpy as np

from wm.grow import GrowConfig, finish, grow


def k_value(tag, n):
    return {"one": 1, "three": 3, "quarter": max(2, n // 4), "all": 0}[tag]


def configs(grid):
    if grid == "main":  # simultaneity x adjacency x direction, equal rates, random picks
        for k, adj, d in itertools.product(["one", "three", "quarter", "all"],
                                           ["push", "local", "nopush", "independent"],
                                           ["iso", "side", "free", "corner"]):
            yield dict(k=k, adjacency=adj, direction=d, rate="equal", pick="random")
    elif grid == "rate":  # rate rules x group picks, push and local adjacency
        for k, pick, rate, adj in itertools.product(["three", "quarter"], ["random", "smallest", "cluster", "void"],
                                                    ["equal", "random", "contacts", "localmin"],
                                                    ["push", "local"]):
            yield dict(k=k, adjacency=adj, direction="iso", rate=rate, pick=pick)
        for rate, adj in itertools.product(["random", "contacts", "localmin"], ["push", "local", "independent"]):
            yield dict(k="all", adjacency=adj, direction="iso", rate=rate, pick="random")
        for k, d in itertools.product(["one", "three", "quarter"], ["iso", "free"]):  # gap-guided growth
            yield dict(k=k, adjacency="push", direction=d, rate="equal", pick="void")
    elif grid == "band":  # how unequal sizes may get, plus axis-aligned seeds
        for band, k, angles in itertools.product([0.005, 0.02, 0.05, 0.2, 1.0, float("inf")],
                                                 ["one", "quarter", "all"], ["random", "zero"]):
            yield dict(k=k, adjacency="push", direction="iso", rate="equal", pick="random", band=band,
                       angles=angles)
    elif grid == "control":  # no growth: random gas at density ~0.6 or ~0.3, straight to squeeze/compact
        for dens in (0.3, 0.6):
            yield dict(k="none", adjacency="-", direction="-", rate="-", pick="-", density=dens)
    else:
        raise ValueError(grid)


def run(job):
    n, seed, c = job
    c = dict(c)
    extra = {}
    if c["k"] == "none":
        extra = dict(max_steps=0, s0=math.sqrt(c["density"] / n))
    cfg = GrowConfig(**extra, k=k_value(c["k"], n) if c["k"] != "none" else 0, pick=c["pick"], direction=c["direction"], rate=c["rate"],
                     adjacency=c["adjacency"], band=c.get("band", 0.05), angles=c.get("angles", "random"),
                     seed=seed)
    t0 = time.time()
    r = grow(n, cfg)
    t1 = time.time()
    sq = finish(r, "squeeze")
    cp = finish(r, "compact")
    out = dict(c, n=n, seed=seed, steps=r["steps"], fails=r["fails"], revivals=r["revivals"],
               direct=r["side_direct"], squeeze=sq[3] if sq else None, compact=cp[3] if cp else None,
               spread=float(r["s"].max() / r["s"].min()), t_grow=t1 - t0, t_finish=time.time() - t1)
    best = min(v for v in (out["squeeze"], out["compact"]) if v is not None)
    out["best"] = best
    if best < BEST.get(n, 1e9) - 1e-9:
        xyz = sq if sq and sq[3] == best else cp
        out["xyt"] = [list(map(float, a)) for a in xyz[:3]]
    return out


BEST = {}


def init(best_path):
    global BEST
    os.environ.setdefault("NUMBA_NUM_THREADS", "1")
    if best_path:
        BEST = {int(k): v for k, v in json.load(open(best_path)).items()}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--grid", default="main")
    p.add_argument("--ns", default="10,15,20,27,30,37,40,50")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--best", default="")
    p.add_argument("--out", default="grow.jsonl")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    a = p.parse_args()
    ns = [int(v) for v in a.ns.split(",")]
    seeds = [int(v) for v in a.seeds.split(",")]
    jobs = [(n, s, c) for c in configs(a.grid) for n in ns for s in seeds]
    done_keys = set()
    if os.path.exists(a.out):  # resume: skip runs already in the output file
        for line in open(a.out):
            r = json.loads(line)
            done_keys.add(json.dumps([r["n"], r["seed"], {k: r[k] for k in jobs[0][2] if k in r}], sort_keys=True))
    jobs = [j for j in jobs if json.dumps([j[0], j[1], j[2]], sort_keys=True) not in done_keys]
    jobs.sort(key=lambda j: -j[0] * (j[0] if j[2]["k"] == "one" else 1))  # long jobs first
    print(f"{len(jobs)} runs", flush=True)
    done = 0
    t0 = time.time()
    with Pool(a.workers, initializer=init, initargs=(a.best,)) as pool, open(a.out, "a") as f:
        for out in pool.imap_unordered(run, jobs):
            f.write(json.dumps(out) + "\n")
            f.flush()
            done += 1
            if done % 50 == 0:
                print(f"{done}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
            if "xyt" in out:
                print("WIN", out["n"], out["best"], {k: out[k] for k in ("k", "adjacency", "direction", "rate")},
                      flush=True)
