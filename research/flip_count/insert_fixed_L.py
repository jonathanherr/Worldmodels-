"""Flipped search: fix the box side L, try to fit one more square than our file holds there.

For a target n+1 we fix L just under our current side s(n+1) and start from our packing of n
(scaled up to L, so it has slack). One unit square goes into each of the largest holes (found by a
distance transform of a raster of the packing) at a few angles, then FIRE relaxes the overlap
energy at that fixed L. Energy 0 means n+1 squares fit in a box smaller than s(n+1): a win.
"""
import json, sys, time
import numpy as np, pandas as pd
from scipy import ndimage
from multiprocessing import Pool
from wm import pack_opt

CSV = sys.argv[1]
OUT = sys.argv[2]
TARGETS = [int(v) for v in sys.argv[3].split(",")] if len(sys.argv) > 3 else None
import os
FRACS = tuple(float(v) for v in os.environ.get("FRACS", "0.999,0.99,0.95").split(","))
NHOLES = int(os.environ.get("NHOLES", "12"))
ANGLES = (0.0, np.pi / 4, np.pi / 8, -np.pi / 8)[:int(os.environ.get("NANG", "4"))]

df = pd.read_csv(CSV)
G = {int(n): (g.x.values.astype(float), g.y.values.astype(float), g.angle.values.astype(float)) for n, g in df.groupby("n")}

def side(x, y, t):
    h = 0.5 * (np.abs(np.cos(t)) + np.abs(np.sin(t)))
    return max((x + h).max() - (x - h).min(), (y + h).max() - (y - h).min()), h

def holes(x, y, t, L, k=12, res=0.04):
    m = int(np.ceil(L / res))
    xs = (np.arange(m) + 0.5) * res
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    occ = np.zeros((m, m), bool)
    for xi, yi, ti in zip(x, y, t):
        c, s = np.cos(ti), np.sin(ti)
        r = 0.71
        i0, i1 = max(0, int((xi - r) / res)), min(m, int((xi + r) / res) + 1)
        j0, j1 = max(0, int((yi - r) / res)), min(m, int((yi + r) / res) + 1)
        dx = X[i0:i1, j0:j1] - xi; dy = Y[i0:i1, j0:j1] - yi
        u = dx * c + dy * s; v = -dx * s + dy * c
        occ[i0:i1, j0:j1] |= (np.abs(u) <= 0.5) & (np.abs(v) <= 0.5)
    pad = np.pad(~occ, 1, constant_values=False)
    d = ndimage.distance_transform_edt(pad)[1:-1, 1:-1] * res
    out = []
    dd = d.copy()
    for _ in range(k):
        i, j = np.unravel_index(np.argmax(dd), dd.shape)
        if dd[i, j] <= 0: break
        out.append((xs[i], xs[j], d[i, j]))
        r = int(1.0 / res)
        dd[max(0, i - r):i + r, max(0, j - r):j + r] = 0
    return out

def attempt(n1):
    n = n1 - 1
    x, y, t = G[n]
    sn, _ = side(x, y, t)
    s1, _ = side(*G[n1])
    h = 0.5 * (np.abs(np.cos(t)) + np.abs(np.sin(t)))
    x = x - (x - h).min(); y = y - (y - h).min()
    best = None
    t0 = time.time()
    for frac in FRACS:  # box between s(n) and s(n+1), mostly just under s(n+1)
        L = sn + frac * (s1 - sn)
        if L - sn < 1e-6: continue
        k = L / sn
        xs, ys = x * k, y * k
        for hx, hy, r in holes(xs, ys, t, L, k=NHOLES):
            for ang in ANGLES:
                X = np.append(xs, hx); Y = np.append(ys, hy); T = np.append(t, ang)
                X2, Y2, T2, e = pack_opt.relax_fire(X, Y, T, L, pack_opt.MARGIN, 20000)
                if best is None or e < best[0]:
                    best = (e, L, frac, hx, hy, r, ang)
                if pack_opt.feasible(X2, Y2, T2, L):
                    return dict(n=n1, win=True, L=L, old=s1, frac=frac, x=X2.tolist(), y=Y2.tolist(), t=T2.tolist())
    return dict(n=n1, win=False, old=s1, sn=sn, best_e=best[0] if best else None, best=best[1:] if best else None, sec=time.time() - t0)

if __name__ == "__main__":
    pack_opt.relax_fire(np.zeros(2), np.array([0.5, 2.0]), np.zeros(2), 3.0, 1e-9, 5)
    with Pool(4) as p:
        res = []
        for r in p.imap_unordered(attempt, TARGETS):
            res.append(r)
            print(r["n"], r["win"], r.get("best_e"), flush=True)
            json.dump(res, open(OUT, "w"))
