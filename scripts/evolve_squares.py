"""Neuroevolution solver: evolve a packing net with CMA-ES, refine, and offer results to the store.

Examples:
    # One shared net for n = 11..20, 80 generations, results into the shared store
    SQUARES_BEST_DIR=/mnt/project-files/squares/best python -m scripts.evolve_squares --n 11-20 --gens 80

    # Separate net per n (more compute, usually a little better per size)
    python -m scripts.evolve_squares --n 11-14 --per-n

    # Control run: random layouts with the same budget, to check the net is earning its keep
    python -m scripts.evolve_squares --n 11-14 --baseline

    # Let the net deform the stored best packings (e.g. construction results) instead of the grid
    python -m scripts.evolve_squares --n 37,50 --per-n --from-store

    # Only basin-hop what is already stored (e.g. construction packings), no evolution
    python -m scripts.evolve_squares --n 30-40 --refine-only

Several copies can run at once (different --n ranges or --seed) against one SQUARES_BEST_DIR;
the store keeps whichever certified packing is shortest.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from wm import best_store
from wm.neuroevo import NetShape, evolve, random_genome_baseline, refine
from wm.pack_opt import squeeze
from wm.squares import polish


def parse_ns(spec: str) -> list[int]:
    ns = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        ns.extend(range(int(a), int(b or a) + 1))
    return ns


def finish(n, side, pk, args, source):
    """Refine a squeezed packing, polish it with a gap, and offer it to the store."""
    if pk is None:
        return None
    x, y, t = pk
    if args.refine_iters:
        x, y, t, side = refine(x, y, t, side, iters=args.refine_iters, seed=args.seed)
    px, py, pd, rep = polish(x, y, np.rad2deg(t), gap=1e-9)
    res = best_store.offer(n, px, py, pd, source) if rep["valid"] else {"accepted": False, "reason": "polish"}
    print(f"n={n:4d}  side {rep['side']:.6f}  stored before {res.get('previous', float('nan')):.6f}  "
          f"grid {math.ceil(math.sqrt(n))}  -> {'KEPT' if res['accepted'] else res['reason']}", flush=True)
    return {"n": n, "side": rep["side"], "accepted": res["accepted"], "source": source}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", required=True, help="sizes, e.g. 11-20 or 5,10,11")
    p.add_argument("--gens", type=int, default=60)
    p.add_argument("--pop", type=int, default=None)
    p.add_argument("--sigma", type=float, default=0.5)
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--rounds", type=int, default=16, help="squeeze rounds per fitness evaluation")
    p.add_argument("--refine-iters", type=int, default=60, help="basin-hopping steps on each final packing")
    p.add_argument("--per-n", action="store_true", help="evolve a separate net for each n")
    p.add_argument("--baseline", action="store_true", help="random layouts with the same budget instead of a net")
    p.add_argument("--from-store", action="store_true",
                   help="have the net deform the stored best packing for each n instead of the grid")
    p.add_argument("--refine-only", action="store_true", help="only basin-hop the stored packings")
    p.add_argument("--out", default="runs/neuroevo")
    args = p.parse_args()

    ns = parse_ns(args.n)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    summary = []
    if args.refine_only:
        for n in ns:
            stored = best_store.load_xyd(n)
            if stored is None:
                print(f"n={n}: nothing stored, skipping")
                continue
            x, y, d = stored
            r = squeeze(x, y, np.deg2rad(d), best_store.best_side(n), rounds=20, step=0.002)
            if r is not None:
                summary.append(finish(n, r[3], r[:3], args, "neuroevo:refine"))
    elif args.baseline:
        # Same number of squeezes as the CMA-ES run would spend on each n.
        evals = args.gens * (args.pop or 4 + int(3 * math.log(NetShape(hidden=args.hidden).size)))
        for n in ns:
            side, pk = random_genome_baseline(n, evals, seed=args.seed, rounds=args.rounds)
            summary.append(finish(n, side, pk, args, "neuroevo:random-baseline"))
    else:
        bases = {}
        if args.from_store:
            for n in ns:
                stored = best_store.load_xyd(n)
                if stored is not None:
                    bases[n] = (stored[0], stored[1], np.deg2rad(stored[2]), best_store.best_side(n))
        groups = [[n] for n in ns] if args.per_n else [ns]
        for group in groups:
            genome, best, shape = evolve(group, gens=args.gens, popsize=args.pop, sigma=args.sigma,
                                         hidden=args.hidden, seed=args.seed, workers=args.workers,
                                         rounds=args.rounds, bases=bases)
            tag = f"{group[0]}-{group[-1]}"
            np.save(out / f"genome_{tag}_h{shape.hidden}_s{args.seed}.npy", genome)
            for n in group:
                summary.append(finish(n, *best[n], args, "neuroevo:cma" + ("-store" if n in bases else "")))
    summary = [s for s in summary if s]
    print(f"done in {time.time() - t0:.0f}s; kept {sum(s['accepted'] for s in summary)} of {len(summary)}; "
          f"store total now {best_store.total_score():.4f}")
    (out / f"summary_{args.n}_s{args.seed}.json").write_text(json.dumps(summary, indent=1))
