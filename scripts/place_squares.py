"""Evolve a sequential placement policy (wm/placer.py) per n, refine the best packing, offer it to the store.

    SQUARES_BEST_DIR=/mnt/project-files/squares/best python -m scripts.place_squares --n 37 --target 6.60

Without --target the box is the stored best side minus --shrink, so the policy is asked to beat it.
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
    p.add_argument("--out", default="runs/placer")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary, t0 = [], time.time()
    for n in parse_ns(args.n):
        L = args.target or best_store.best_side(n) - args.shrink
        print(f"n={n}: target box {L:.4f} (stored {best_store.best_side(n):.4f})", flush=True)
        genome, (side, pk) = placer.evolve(n, L, gens=args.gens, popsize=args.pop, hidden=args.hidden,
                                           step=args.step, tol=args.tol, seed=args.seed, workers=args.workers)
        np.save(out / f"placer_{n}_s{args.seed}.npy", genome)
        summary.append(finish(n, side, pk, SimpleNamespace(refine_iters=args.refine_iters, seed=args.seed),
                              "neuroevo:placer"))
    summary = [s for s in summary if s]
    print(f"done in {time.time() - t0:.0f}s; kept {sum(s['accepted'] for s in summary)} of {len(summary)}; "
          f"store total now {best_store.total_score():.4f}", flush=True)
    (out / f"summary_{args.n}_s{args.seed}.json").write_text(json.dumps(summary, indent=1))
