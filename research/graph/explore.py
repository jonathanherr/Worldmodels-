"""Best-first search over the contact-structure graph of one n.

  python explore.py CSV N --minutes 25 --top 10 --procs 4 --out run.jsonl

Each expanded node tries flip(+1), flip(-1), swap (tilts differ) and break(0.3) on its `top`
most loaded contacts; every child is released and SLP-polished, hashed, and queued by L.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import time
from multiprocessing import Pool

import numpy as np

import contact_graph as C
import slp as S


def child(args):
    x, y, t, a, b, k, kind = args
    if kind == "flip+":
        f, g = (a, b, (k + 1) % 4), 0.0
    elif kind == "flip-":
        f, g = (a, b, (k + 3) % 4), 0.0
    elif kind == "swap":
        f, g = (a, b, (k + 2) % 4), 0.0
    else:
        f, g = (a, b, k), 0.3
    try:
        X, Y, T, _, fv = C.run_forced(x, y, t, f, fgap=g)
        X, Y, T, L, v, _ = S.slp(X, Y, T, iters=400)
        if v > 1e-9:
            return None
        d, _ = C.describe(X, Y, T, tol=1e-7)
        return dict(L=float(L), hash=d["hash"], contacts=d["contacts"], stressed=d["stressed"],
                    move=[int(a), int(b), int(k), kind], x=X, y=Y, t=T)
    except Exception as e:  # noqa: BLE001
        return dict(err=str(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv"); ap.add_argument("n", type=int)
    ap.add_argument("--minutes", type=float, default=25); ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--procs", type=int, default=4); ap.add_argument("--out", default=None)
    ap.add_argument("--best", default=None, help="npz to write the best packing found")
    ap.add_argument("--key", choices=["hash", "L"], default="hash",
                    help="node identity: contact-graph WL hash, or side rounded to 1e-7 (one node per L level)")
    a = ap.parse_args()
    x, y, t = C.load_csv_n(a.csv, a.n)
    x, y, t, L0, v, _ = S.slp(x, y, t, iters=300)
    d0, _ = C.describe(x, y, t, tol=1e-7)
    key = (lambda h, L: h) if a.key == "hash" else (lambda h, L: round(L, 7))
    seen = {key(d0["hash"], L0): L0}
    heap = [(L0, 0, d0["hash"], x, y, t, 0)]
    out = open(a.out or f"graph_{a.n}.jsonl", "w")
    out.write(json.dumps(dict(root=True, n=a.n, L=L0, **{k: d0[k] for k in ("contacts", "stressed", "dof", "hash")})) + "\n")
    best = (L0, x, y, t)
    t_end = time.time() + 60 * a.minutes
    uid, expanded, edges, same_L = 1, 0, 0, 0
    with Pool(a.procs) as pool:
        while heap and time.time() < t_end:
            L, _, h, x, y, t, depth = heapq.heappop(heap)
            ca, cb, ck, load, _ = C.contacts(x, y, t, tol=1e-7)
            order = np.argsort(-load)[: a.top]
            jobs = []
            for i in order:
                u, w, k = int(ca[i]), int(cb[i]), int(ck[i])
                kinds = ["flip+", "flip-", "break"]
                dt = abs(((t[u] - t[w]) % (math.pi / 2)))
                if min(dt, math.pi / 2 - dt) > 0.05:
                    kinds.append("swap")
                jobs += [(x, y, t, u, w, k, kd) for kd in kinds]
            expanded += 1
            for r in pool.imap_unordered(child, jobs):
                edges += 1
                if not r or "err" in r:
                    continue
                rec = {k: r[k] for k in ("L", "hash", "contacts", "stressed", "move")}
                rec.update(parent=h, depth=depth + 1, new=key(r["hash"], r["L"]) not in seen)
                out.write(json.dumps(rec) + "\n"); out.flush()
                if abs(r["L"] - L0) < 1e-9:
                    same_L += 1
                if r["L"] < best[0] - 1e-10:
                    best = (r["L"], r["x"], r["y"], r["t"])
                    print(f"  BETTER L {r['L']:.9f} (root {L0:.9f}) move {r['move']} depth {depth+1}", flush=True)
                if key(r["hash"], r["L"]) not in seen:
                    seen[key(r["hash"], r["L"])] = r["L"]
                    heapq.heappush(heap, (r["L"], uid, r["hash"], r["x"], r["y"], r["t"], depth + 1)); uid += 1
            print(f"n {a.n} expanded {expanded} (L {L:.6f} depth {depth}) edges {edges} nodes {len(seen)} "
                  f"best {best[0]:.9f} root {L0:.9f} queue-min {heap[0][0] if heap else float('nan'):.6f}", flush=True)
    out.write(json.dumps(dict(summary=True, n=a.n, L0=L0, best=best[0], nodes=len(seen), edges=edges,
                              expanded=expanded, same_L=same_L)) + "\n")
    if a.best:
        np.savez(a.best, x=best[1], y=best[2], t=best[3], L=best[0])


if __name__ == "__main__":
    main()
