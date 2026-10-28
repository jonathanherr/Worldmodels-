"""Sweep the one-agent-per-square packer (wm/agentpack.py) over n and seeds, with ablations.

    python -m scripts.agent_squares --n 5-40 --seeds 4 --modes all,none,hop,slide,coalition \
        --out /mnt/project-files/squares/agents/run1.json [--actors 10 --rounds 30] [--offer]

Mode `all` uses every behaviour, `none` the same shrink schedule with gradient relaxation only,
`control` the random gas -> compact_squeeze baseline. Each result is compared with the shared
store ($SQUARES_BEST_DIR) and the public record store; --offer sends wins to the shared store.
"""

import argparse
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

RECORDS = Path("/mnt/project-files/squares/public-certs/store")
MODES = {"all": ("hop", "slide", "coalition"), "none": (), "hop": ("hop",),
         "slide": ("slide",), "coalition": ("coalition",)}


def parse_ns(spec):
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += range(int(a), int(b) + 1)
        else:
            out.append(int(part))
    return out


def job(args):
    n, seed, mode, actors, rounds = args
    from wm.agentpack import agent_pack, control_pack
    from wm.squares import official_check

    t0 = time.time()
    if mode == "control":
        x, y, t, L = control_pack(n, seed=seed)
        stats = {}
    else:
        x, y, t, L, stats = agent_pack(n, seed=seed, use=MODES[mode], actors=actors, rounds=rounds)
    rows = [{"id": k, "n": n, "square_id": k, "x": float(x[k]), "y": float(y[k]),
             "angle": float(t[k])} for k in range(n)]
    chk = official_check(rows)
    return {"n": n, "seed": seed, "mode": mode, "L": float(L), "valid": chk["valid"],
            "official": chk["side"], "secs": time.time() - t0, "stats": stats,
            "x": x.tolist(), "y": y.tolist(), "t": t.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", default="5-30")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--modes", default="all,none,control")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--offer", action="store_true")
    ap.add_argument("--actors", type=int, default=3, help="stressed agents that act per round")
    ap.add_argument("--rounds", type=int, default=12, help="negotiation rounds per box size")
    a = ap.parse_args()

    from wm import best_store

    jobs = [(n, s, m, a.actors, a.rounds) for n in parse_ns(a.n) for s in range(a.seeds) for m in a.modes.split(",")]
    jobs.sort(key=lambda j: -j[0])  # big n first for load balance
    results = []
    with Pool(a.workers) as pool:
        for r in pool.imap_unordered(job, jobs):
            r["store"] = best_store.best_side(r["n"])
            r["record"] = best_store.best_side(r["n"], RECORDS)
            r["grid"] = best_store.grid_side(r["n"])
            results.append(r)
            print(f"n={r['n']:4d} {r['mode']:9s} s{r['seed']} L={r['L']:.5f} grid={r['grid']:.0f} "
                  f"store={r['store']:.5f} rec={r['record']:.5f} {r['secs']:.0f}s", flush=True)
            if a.offer and r["valid"] and r["official"] < r["store"] - 1e-9:
                res = best_store.offer(r["n"], r["x"], r["y"], np.rad2deg(r["t"]), f"agents:{r['mode']}")
                print("   offer", res, flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(results))


if __name__ == "__main__":
    main()
