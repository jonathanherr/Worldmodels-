"""Evolve a sequential placement policy (wm/placer.py) per n, refine the best packing, offer it to the store.

    SQUARES_BEST_DIR=/mnt/project-files/squares/best python -m scripts.place_squares --n 37 --target 6.60

Without --target the box is the stored best side minus --shrink, so the policy is asked to beat it.
With --curriculum the stored packing guides the policy and the target shrinks in stages:

    python -m scripts.place_squares --n 37,50 --curriculum --gens 15

"""

import argparse
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts.evolve_squares import finish, parse_ns
from wm import best_store, placer

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", required=True)
    p.add_argument("--target", type=float, default=None, help="target box side (one n only)")
    p.add_argument("--shrink", type=float, default=0.02, help="target = stored side - shrink")
    p.add_argument("--gens", type=int, default=80)
    p.add_argument("--pop", type=int, default=None)
    p.add_argument("--hidden", type=int, default=8)
    p.add_argument("--step", type=float, default=0.1, help="candidate grid spacing")
    p.add_argument("--tol", type=float, default=0.02, help="overlap a placement may have (the squeeze removes it)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--refine-iters", type=int, default=20)
    p.add_argument("--curriculum", action="store_true",
                   help="guide the policy with the stored packing and shrink the target in stages (--gens per stage)")
    p.add_argument("--shrinks", default="0,0.01,0.02,0.04,0.06,0.08", help="curriculum: target = stored side - each")
    p.add_argument("--out", default="runs/placer")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary, t0 = [], time.time()
    for n in parse_ns(args.n):
        L = args.target or best_store.best_side(n) - args.shrink
        print(f"n={n}: target box {L:.4f} (stored {best_store.best_side(n):.4f})", flush=True)
        kw = dict(popsize=args.pop, hidden=args.hidden, step=args.step, tol=args.tol, seed=args.seed,
                  workers=args.workers)
        if args.curriculum:
            stored = best_store.load_xyd(n)
            if stored is None:
                print(f"n={n}: nothing stored to start from, skipping", flush=True)
                continue
            guide = (stored[0], stored[1], np.deg2rad(stored[2]), best_store.best_side(n))
            shrinks = [float(v) for v in args.shrinks.split(",")]
            side, pk = placer.curriculum(n, guide, shrinks=shrinks, gens=args.gens, **kw)
            source = "neuroevo:placer-curriculum"
        else:
            genome, (side, pk), _ = placer.evolve(n, L, gens=args.gens, **kw)
            np.save(out / f"placer_{n}_s{args.seed}.npy", genome)
            source = "neuroevo:placer"
        summary.append(finish(n, side, pk, SimpleNamespace(refine_iters=args.refine_iters, seed=args.seed), source))
    summary = [s for s in summary if s]
    print(f"done in {time.time() - t0:.0f}s; kept {sum(s['accepted'] for s in summary)} of {len(summary)}; "
          f"store total now {best_store.total_score():.4f}", flush=True)
    (out / f"summary_{args.n}_s{args.seed}.json").write_text(json.dumps(summary, indent=1))
