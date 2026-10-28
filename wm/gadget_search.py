"""Gadget search: score a new fault-line gadget by what it does for the whole submission.

A gadget (see wm.fault_dp) is m = w*h + d unit squares in the rectangle [0, w + f] x [0, h + f].
Its own number is f, but f alone says little: a gadget only pays off if some chain of gadgets
along a K x K staircase reaches K + max(f) below the side we already have for n = K^2 + sum(d).
So this module keeps the gadget pool of a base submission (every corner crop of every base
packing and of the big public certificates) and scores a candidate by rerunning the DP with the
candidate's crops added:

    pool = build_pool(base_csv, certs_dir)          # ~2 min on 4 CPUs, cached to a pickle
    r = score_packing(pool, squares)                # squares: (x, y, angle_rad) rows
    r["gain"]                                       # predicted score points over the base file
    r["wins"]                                       # {n: (old side, new side, chain)}

`verify_wins` builds the winning chains and checks them with the official metric. The DP takes
~20 s; the prediction is exact for chains, so verification only guards against tip overlaps.

`value_table` asks the reverse question: if a gadget (w, h, d) existed at f, how many points
would the DP gain? It inserts a hypothetical gadget and reruns the DP, which tells agents which
(w, h, d) are worth designing before they spend time on one.
"""

from __future__ import annotations

import math
import pickle
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from . import fault_dp as fd

ROUNDS = 4  # extra gadgets per chain, as in scripts/fault_dp.py
_DONORS: dict = {}


def _extract(i):
    return fd.extract(i, _DONORS[i])


def build_pool(base_csv, certs_dir=None, cert_min=1001, workers=4, cache=None) -> dict:
    """Donors, gadget groups and base sides for a submission CSV (angles in radians)."""
    import pandas as pd

    if cache and Path(cache).exists():
        return pickle.loads(Path(cache).read_bytes())
    if base_csv is None:
        raise FileNotFoundError(f"no gadget pool at {cache}; build it with `python -m scripts.gadget_search pool`")
    t0 = time.time()
    df = pd.read_csv(base_csv)
    base = fd.load_csv_donors(df)
    sides = {n: fd.side_of(p) for n, p in base.items()}
    donors = list(base.values())
    if certs_dir:
        for path in sorted(Path(certs_dir).glob("n*.cert")):
            if int(path.stem[1:]) >= cert_min:
                donors.append(fd.load_cert(path))
    _DONORS.clear()
    _DONORS.update(enumerate(donors))
    with Pool(workers) as p:
        gadgets = [g for part in p.imap_unordered(_extract, range(len(donors)), chunksize=2) for g in part]
    pool = {"donors": donors, "groups": fd.build_groups(gadgets), "sides": sides,
            "base": str(base_csv), "seconds": time.time() - t0}
    baseline(pool)
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        Path(cache).write_bytes(pickle.dumps(pool))
    return pool


def _merge(groups, extra):
    """groups with the extra gadgets folded in (copies only the touched entries)."""
    out = dict(groups)
    for key, (fv, src) in fd.build_groups(extra).items():
        if key in out:
            fv0, src0 = out[key]
            fv1, src1 = fv0.copy(), dict(src0)
            better = fv < fv1
            fv1[better] = fv[better]
            src1.update({d: src[d] for d in np.nonzero(better)[0]})
            out[key] = (fv1, src1)
        else:
            out[key] = (fv, src)
    return out


def dp_wins(groups, sides, rounds=ROUNDS, kmin=4, max_n=1000):
    """{n: (old side, new side, K, D, f, r, b, W, H)} for every n a chain beats, and the layers."""
    layers = fd.run_dp(groups, rounds)
    wins = {}
    for K in range(kmin, 32):
        for D in range(1, min(fd.MAX_D, max_n - K * K) + 1):
            res = fd.best_chains(layers, K, D)
            if res is None:
                continue
            f, r, b, W, H = res
            for n in range(K * K + 1, K * K + D + 1):
                if K + f < sides[n] - 1e-9 and (n not in wins or K + f < wins[n][1]):
                    wins[n] = (sides[n], K + f, K, D, f, r, b, W, H)
    return wins, layers


def gain_of(wins):
    return sum((old - new) / math.sqrt(n) for n, (old, new, *_) in wins.items())


def baseline(pool):
    """Wins the pool already has on its own base file (about 0 once fault_dp has been applied)."""
    if "baseline" not in pool:
        wins, _ = dp_wins(pool["groups"], pool["sides"])
        pool["baseline"] = {n: w[1] for n, w in wins.items()}
    return pool["baseline"]


def gadget_f(sq, w, h):
    """Bloat f of squares (x, y, angle_rad) in the w x h corner rectangle, after normalizing."""
    p = fd.normalize(sq)
    he = fd.half_ext(p[:, 2])
    return max(0.0, float((p[:, 0] + he).max() - w), float((p[:, 1] + he).max() - h))


def best_known(pool, w, h, d):
    """Smallest f the pool has for (w, h, d), over tip variants (either orientation)."""
    best = math.inf
    for (gw, gh, tl, br), (fv, _) in pool["groups"].items():
        if (gw, gh) in ((w, h), (h, w)) and d < len(fv):
            best = min(best, float(fv[d]))
    return best


def known_keys(pool, wmax=31):
    """{(w, h, d) with w <= h: best f in the pool} for every gadget shape the pool has."""
    best = {}
    for (gw, gh, tl, br), (fv, _) in pool["groups"].items():
        w, h = min(gw, gh), max(gw, gh)
        if h > wmax:
            continue
        for d in np.nonzero(np.isfinite(fv))[0]:
            k = (w, h, int(d))
            best[k] = min(best.get(k, math.inf), float(fv[d]))
    return best


def score_packing(pool, sq):
    """Add every corner crop of the packing sq (x, y, angle_rad rows) to the pool and rerun the DP.

    Returns predicted gain over the base file (minus what the pool alone already gains), the
    winning n, and the candidate's own gadgets that the winning chains use.
    """
    p = fd.normalize(np.asarray(sq, dtype=float))
    did = len(pool["donors"])
    extra = fd.extract(did, p)
    groups = _merge(pool["groups"], extra)
    wins, layers = dp_wins(groups, pool["sides"])
    base = baseline(pool)
    new = {n: w for n, w in wins.items() if n not in base or w[1] < base[n] - 1e-12}
    gain = sum((w[0] - w[1]) / math.sqrt(n) - ((w[0] - base[n]) / math.sqrt(n) if n in base else 0.0)
               for n, w in new.items())
    return {"gain": gain, "wins": new, "n_gadgets": len(extra), "groups": groups, "layers": layers,
            "donors": pool["donors"] + [p]}


def verify_wins(r, rows_out=None, workers=4):
    """Build each winning chain and check it with the official metric. {n: side} of the passes."""
    jobs = []
    for n, (old, new, K, D, f, rr, b, W, H) in r["wins"].items():
        chain = fd.backtrack(r["layers"], r["groups"], rr, b, W, H, D, f)
        if chain is None:
            continue
        built, _ = fd.compose(chain, r["donors"], K)
        jobs.append((n, fd.to_rows(n, fd.drop_to(built, n))))
    ok = {}
    with Pool(workers) as p:
        for (n, rows), (side, reason) in zip(jobs, p.starmap(fd.check_rows, jobs)):
            if side is not None and side < r["wins"][n][0] - 1e-12:
                ok[n] = side
                if rows_out is not None:
                    rows_out[n] = rows
    return ok


def value_table(pool, keys, f, tips=(True, True)):
    """[(gain, w, h, d)] for hypothetical gadgets (w, h, d) at bloat f (a number or one per key)."""
    base = baseline(pool)
    out = []
    fs = f if isinstance(f, (list, tuple)) else [f] * len(keys)
    for (w, h, d), f in zip(keys, fs):
        g = fd.Gadget(w, h, d, float(f), tips[0], tips[1], -1, 0)
        groups = _merge(pool["groups"], [g, fd.Gadget(h, w, d, float(f), tips[1], tips[0], -1, 0)])
        wins, _ = dp_wins(groups, pool["sides"])
        gain = sum((w_[0] - w_[1]) / math.sqrt(n) - ((w_[0] - base[n]) / math.sqrt(n) if n in base else 0.0)
                   for n, w_ in wins.items() if n not in base or w_[1] < base[n] - 1e-12)
        out.append((gain, w, h, d))
    return sorted(out, reverse=True)
