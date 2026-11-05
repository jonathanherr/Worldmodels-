"""Polish fault-DP gadgets as rectangles: minimise f for a w x h gadget holding w*h+d squares."""
import sys, json, time, numpy as np
import os as _os; _H = _os.path.dirname(_os.path.abspath(__file__)); sys.path.insert(0, _H); sys.path.insert(0, _os.path.dirname(_os.path.dirname(_H)))
from slp import slp, side
from wm import fault_dp as fd

def best_gadget(p, w, h, d):
    best = None
    for g in fd.extract(0, p, max_wh=max(w, h)):
        if (g.w, g.h, g.d) == (w, h, d) and (best is None or g.f < best.f):
            best = g
    return best

def hop(q, w, h, rounds=20, sigma=0.03, seed=0, verbose=True):
    rng = np.random.default_rng(seed)
    x, y, t, L, v, L0 = slp(q[:, 0], q[:, 1], q[:, 2], iters=200, ex=w, ey=h)
    best = (L, np.c_[x, y, t]); 
    if verbose: print(f'  polish f {L0:.6f} -> {L:.6f} v {v:.1e}', flush=True)
    for r in range(rounds):
        b = best[1].copy()
        k = rng.integers(len(b)); c = b[k, :2]
        dist = np.hypot(*(b[:, :2] - c).T)
        m = dist < rng.uniform(1.5, 4)
        b[m, :2] += rng.normal(0, sigma, (m.sum(), 2)); b[m, 2] += rng.normal(0, sigma, m.sum())
        x, y, t, L, v, _ = slp(b[:, 0], b[:, 1], b[:, 2], iters=250, ex=w, ey=h)
        if v < 1e-10 and L < best[0] - 1e-9:
            best = (L, np.c_[x, y, t])
            if verbose: print(f'  hop {r} f {L:.6f}', flush=True)
    return best

if __name__ == '__main__':
    src, n, w, h, d = sys.argv[1], sys.argv[2], *map(int, sys.argv[3:6])
    rows = json.load(open(src))[n]
    p = np.array([[float(r['x']), float(r['y']), float(r['angle'])] for r in rows])
    g = best_gadget(p, w, h, d)
    q = fd.gadget_squares(p, g)
    print('gadget', g)
    t0 = time.time()
    L, b = hop(q, w, h, rounds=int(sys.argv[6]) if len(sys.argv) > 6 else 20)
    print(f'final f {L:.6f} (was {g.f:.6f}) {time.time()-t0:.0f}s')
    np.save(f'/home/claude/work/gad_{n}_{w}_{h}_{d}.npy', b)
