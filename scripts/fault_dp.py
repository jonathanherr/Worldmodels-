"""Fault-line gadget DP: compose known packings into new ones and keep those that beat a base file.

    python -m scripts.fault_dp base.csv --certs ~/square-packing/certificates/best --out runs/fault

Donors are every packing in the base submission plus every certificate with n > --cert-min
(bigger packings give bigger gadgets). Each candidate is checked with the official metric; the
winners go to <out>/candidates.json and <out>/submission.csv (base with winners swapped in).
"""

import argparse
import json
import math
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from wm import fault_dp as fd

DONORS = {}


def _extract(i):
    return fd.extract(i, DONORS[i], tip=TIP[0])


TIP = [fd.TIP]


def _check(job):
    n, rows = job
    side, reason = fd.check_rows(n, rows)
    return n, side, reason, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base", help="submission CSV to beat (also used as donors)")
    ap.add_argument("--certs", help="franciscouzo/square-packing certificates/best directory")
    ap.add_argument("--cert-min", type=int, default=1001, help="only certificates with n >= this")
    ap.add_argument("--cert-max", type=int, default=100000)
    ap.add_argument("--out", default="runs/fault")
    ap.add_argument("--rounds", type=int, default=4, help="extra gadgets per chain")
    ap.add_argument("--kmin", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tip", default=str(fd.TIP), help="tip box size, or 'own' for each gadget's own f")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    base_df = pd.read_csv(a.base, dtype={"x": str, "y": str, "angle": str})
    num = base_df.assign(x=base_df.x.astype(float), y=base_df.y.astype(float), angle=base_df.angle.astype(float))
    base = fd.load_csv_donors(num)
    sides = {n: fd.side_of(p) for n, p in base.items()}
    donors = list(base.values())
    if a.certs:
        for path in sorted(Path(a.certs).glob("n*.cert")):
            n = int(path.stem[1:])
            if a.cert_min <= n <= a.cert_max:
                donors.append(fd.load_cert(path))
    print(f"{len(donors)} donors loaded ({time.time() - t0:.0f}s)", flush=True)

    DONORS.update(enumerate(donors))
    TIP[0] = None if a.tip == "own" else float(a.tip)
    with Pool(a.workers) as pool:
        gadgets = [g for part in pool.imap_unordered(_extract, range(len(donors)), chunksize=2) for g in part]
    groups = fd.build_groups(gadgets)
    print(f"{len(gadgets)} gadgets in {len(groups)} groups ({time.time() - t0:.0f}s)", flush=True)
    layers = fd.run_dp(groups, a.rounds)
    print(f"DP done ({time.time() - t0:.0f}s)", flush=True)

    jobs, meta = [], {}
    for K in range(a.kmin, 32):
        for D in range(1, min(fd.MAX_D, 1000 - K * K) + 1):
            top = K * K + D
            res = fd.best_chains(layers, K, D)
            if res is None:
                continue
            f, r, b, W, H = res
            targets = [n for n in range(K * K + 1, top + 1) if K + f < sides[n] - 1e-9]
            if not targets:
                continue
            chain = fd.backtrack(layers, groups, r, b, W, H, D, f)
            if chain is None:
                print(f"K={K} D={D}: backtrack failed", flush=True)
                continue
            sq, _ = fd.compose(chain, donors, K)
            if len(sq) != top:
                print(f"K={K} D={D}: built {len(sq)} squares, expected {top}", flush=True)
                continue
            desc = [(g.w, g.h, g.d, mode) for g, mode in chain]
            for n in targets:
                jobs.append((n, fd.to_rows(n, fd.drop_to(sq, n))))
                meta.setdefault(n, []).append({"K": K, "D": D, "f": f, "chain": desc})
    print(f"{len(jobs)} candidate packings to check ({time.time() - t0:.0f}s)", flush=True)

    best = {}
    with Pool(a.workers) as pool:
        for n, side, reason, rows in pool.imap_unordered(_check, jobs):
            if side is None:
                print(f"n={n}: rejected ({reason[:120]})", flush=True)
                continue
            if side < sides[n] - 1e-12 and (n not in best or side < best[n][0]):
                best[n] = (side, rows)
    gain = sum((sides[n] - s) / math.sqrt(n) for n, (s, _) in best.items())
    for n in sorted(best):
        print(f"n={n}: {sides[n]:.6f} -> {best[n][0]:.6f}  {meta[n][0]}", flush=True)
    print(f"{len(best)} n improved, score gain {gain:.6f} ({time.time() - t0:.0f}s)", flush=True)

    cands = {str(n): {"side": s, "meta": meta[n], "x": [r["x"] for r in rows], "y": [r["y"] for r in rows],
                      "angle": [r["angle"] for r in rows]} for n, (s, rows) in best.items()}
    (out / "candidates.json").write_text(json.dumps(cands))
    keep = base_df[~base_df.n.isin(best)]
    new = pd.DataFrame([r for _, rows in best.values() for r in rows])
    sub = pd.concat([keep[["n", "square_id", "x", "y", "angle"]], new[["n", "square_id", "x", "y", "angle"]]])
    sub = sub.sort_values(["n", "square_id"], key=lambda c: c.astype(int)).reset_index(drop=True)
    sub.insert(0, "id", range(len(sub)))
    sub.to_csv(out / "submission.csv", index=False)
    print(f"wrote {out / 'submission.csv'}", flush=True)


if __name__ == "__main__":
    main()
