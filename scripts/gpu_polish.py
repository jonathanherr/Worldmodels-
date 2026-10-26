"""Batched perturb-and-relax polish (wm/gpu_polish.py) over many n, offering wins to the store.

    # On the M5 (MPS): the n where some Kaggle team is known to beat the public records, worst first
    SQUARES_BEST_DIR=... python -m scripts.gpu_polish --targets targets.json --top 100 --minutes 10

    # Or explicit n
    SQUARES_BEST_DIR=... python -m scripts.gpu_polish --n 577-1000 --minutes 3 --batch 512

--targets is a JSON list of {"n", "gap_score"} (see /mnt/project-files/squares/public-certs/targets.json);
n are run in decreasing gap_score. Each result is one JSON line in --out.
"""

import argparse
import json
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from scripts.evolve_squares import parse_ns  # noqa: E402
from wm.gpu_polish import pick_device, polish_n  # noqa: E402


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", help="e.g. 577-1000 or 335,644")
    p.add_argument("--targets", help="JSON list of {n, gap_score}; runs the largest gaps first")
    p.add_argument("--top", type=int, default=50, help="how many target n to run")
    p.add_argument("--minutes", type=float, default=5.0, help="time budget per n")
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--steps", type=int, default=600, help="relaxation steps per round")
    p.add_argument("--device", default="auto", help="auto, mps, cuda or cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="gpu_polish.jsonl")
    args = p.parse_args()

    if args.targets:
        t = sorted(json.load(open(args.targets)), key=lambda o: -o["gap_score"])
        ns = [o["n"] for o in t if o["gap_score"] > 0][: args.top]
    else:
        ns = parse_ns(args.n)
    device = pick_device(args.device)
    print(f"{len(ns)} n on {device}, {args.minutes} min each", flush=True)
    for n in ns:
        t0 = time.time()
        r = polish_n(n, minutes=args.minutes, device=device, batch=args.batch, steps=args.steps,
                     seed=args.seed, log=lambda s: print(s, flush=True))
        r["seconds"] = round(time.time() - t0, 1)
        print(json.dumps(r), flush=True)
        with open(args.out, "a") as f:
            f.write(json.dumps(r) + "\n")
