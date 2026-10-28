"""Gadget search CLI: build the gadget pool of a submission, list the gadgets worth designing, score one.

  python -m scripts.gadget_search pool base.csv --certs ~/square-packing/certificates/best   # ~2 min, once per base file
  python -m scripts.gadget_search value --delta 0.025 --wmax 12       # hypothetical-gadget DP per (w, h, d), ~25 min
  python -m scripts.gadget_search table [--k 30]                       # what the value run found, with best f known
  python -m scripts.gadget_search score packing.npy [--verify]         # any packing (x, y, angle_rad rows) as donor

Pool: $PROGSEARCH_POOL (default /mnt/project-files/squares/gadgets/pool.pkl); the value table is saved
next to it as value.json. Program-search agents use the "gadget" task of scripts.progsearch, which
calls the same scorer.
"""

import argparse
import json
import math
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from wm import gadget_search as gs
from wm import progsearch as ps

_P: dict = {}


def _value(args):
    key, f = args
    return gs.value_table(_P, [key], f)[0]


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pool"); p.add_argument("base"); p.add_argument("--certs"); p.add_argument("--workers", type=int, default=4)
    v = sub.add_parser("value", help="gain if each known (w, h, d) gadget got delta better")
    v.add_argument("--delta", type=float, default=0.025, help="f improvement to price (leaders are ~0.027 ahead)")
    v.add_argument("--wmax", type=int, default=12)
    v.add_argument("--fmin", type=float, default=0.70, help="only keys whose best known f is in [fmin, fmax]")
    v.add_argument("--fmax", type=float, default=0.95)
    v.add_argument("--workers", type=int, default=4)
    t = sub.add_parser("table"); t.add_argument("--k", type=int, default=30)
    s = sub.add_parser("score"); s.add_argument("packing"); s.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    value_path = ps.POOL.with_name("value.json")

    if a.cmd == "pool":
        if ps.POOL.exists():
            ps.POOL.unlink()
        pool = gs.build_pool(a.base, a.certs, workers=a.workers, cache=ps.POOL)
        print(f"{len(pool['donors'])} donors, {len(pool['groups'])} gadget groups, baseline wins {len(pool['baseline'])} "
              f"({pool['seconds']:.0f}s) -> {ps.POOL}")
    elif a.cmd == "value":
        _P.update(ps.gadget_pool())
        known = gs.known_keys(_P, a.wmax)
        jobs = [((w, h, d), f - a.delta) for (w, h, d), f in known.items() if a.fmin <= f <= a.fmax]
        print(f"{len(jobs)} keys, each priced at its best known f - {a.delta}", flush=True)
        with Pool(a.workers) as pool:
            rows = sorted(pool.imap_unordered(_value, jobs), reverse=True)
        value_path.write_text(json.dumps({"delta": a.delta, "rows": rows}))
        print(f"wrote {value_path}")
    elif a.cmd == "table":
        pool = ps.gadget_pool()
        data = json.loads(value_path.read_text())
        print(f"Points the submission would gain if the best known gadget (w, h, d) got {data['delta']} lower f "
              f"(tips free), best first:")
        for gain, w, h, d in data["rows"][:a.k]:
            f = gs.best_known(pool, w, h, d)
            print(f"  w={w:2d} h={h:2d} d={d:2d}  m={w * h + d:3d}  best known f {f:.5f}  "
                  f"gain at f {f - data['delta']:.5f}: {gain:.5f}")
    elif a.cmd == "score":
        pool = ps.gadget_pool()
        r = gs.score_packing(pool, np.load(a.packing))
        print(f"predicted gain {r['gain']:.6f} on {len(r['wins'])} n")
        for n, (old, new, K, D, f, *_) in sorted(r["wins"].items()):
            print(f"  n={n}: {old:.6f} -> {new:.6f}  (K={K} D={D} f={f:.5f})")
        if a.verify and r["wins"]:
            ok = gs.verify_wins(r)
            print(f"verified {len(ok)} n, gain {sum((pool['sides'][n] - s) / math.sqrt(n) for n, s in ok.items()):.6f}")


if __name__ == "__main__":
    main()
