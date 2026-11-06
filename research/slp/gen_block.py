"""Tilted-block gadgets from scratch.

A w x h gadget (box (w+f) x (h+f)) is built as: a parallelogram patch of a unit-square lattice
tilted by theta (rows i0..i1, columns j0..j1 of the lattice, lattice origin o), plus a greedy fill of
axis squares taken from two integer grids (anchored at the lower-left and at the upper-right corner
of the box). The start is checked for overlap, then SLP shrinks f with the count fixed.

    python research/slp/gen_block.py W H --trials 200 --procs 4 --out runs.jsonl --npy-dir gads
"""
import argparse, json, math, os, sys, time
import numpy as np
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from slp import slp, corners

def sq_corners(x, y, t):
    cx, cy, _, _ = corners(np.atleast_1d(x), np.atleast_1d(y), np.atleast_1d(t))
    return cx, cy

def overlaps(P, q, tol=1e-9):
    """Does square q=(x,y,t) overlap any square in P (n,3)? SAT on candidate neighbours."""
    if len(P) == 0:
        return False
    d = np.hypot(P[:, 0] - q[0], P[:, 1] - q[1])
    near = P[d < 1.42]
    if len(near) == 0:
        return False
    qx, qy = sq_corners(q[0], q[1], q[2])
    qx, qy = qx[0], qy[0]
    cx, cy = sq_corners(near[:, 0], near[:, 1], near[:, 2])
    for k in range(len(near)):
        sep = False
        for ang in (q[2], q[2] + math.pi / 2, near[k, 2], near[k, 2] + math.pi / 2):
            u = (math.cos(ang), math.sin(ang))
            a = qx * u[0] + qy * u[1]; b = cx[k] * u[0] + cy[k] * u[1]
            if a.max() <= b.min() + tol or b.max() <= a.min() + tol:
                sep = True; break
        if not sep:
            return True
    return False

def inside(q, W, H):
    x, y = sq_corners(q[0], q[1], q[2])
    return x.min() >= -1e-12 and y.min() >= -1e-12 and x.max() <= W + 1e-12 and y.max() <= H + 1e-12

def build(w, h, f0, theta, ox, oy, i0, i1, j0, j1, rng):
    W, H = w + f0, h + f0
    c, s = math.cos(theta), math.sin(theta)
    u, v = np.array([c, s]), np.array([-s, c])
    P = []
    for i in range(i0, i1 + 1):
        for j in range(j0, j1 + 1):
            p = np.array([ox, oy]) + i * u + j * v
            q = (p[0], p[1], theta)
            if inside(q, W, H):
                P.append(q)
    P = np.array(P).reshape(-1, 3)
    # axis candidates from two grids, in random order biased to the edges
    cand = []
    for gx in range(w + 1):
        for gy in range(h + 1):
            cand.append((gx + 0.5, gy + 0.5, 0.0))
            cand.append((W - gx - 0.5, H - gy - 0.5, 0.0))
            cand.append((gx + 0.5, H - gy - 0.5, 0.0))
            cand.append((W - gx - 0.5, gy + 0.5, 0.0))
    cand = np.array(cand)
    cand = np.unique(np.round(cand, 9), axis=0)
    ok = [k for k in range(len(cand)) if inside(tuple(cand[k]), W, H) and not overlaps(P, tuple(cand[k]))]
    cand = cand[ok]
    # exact max independent set of axis squares (MILP); two axis squares conflict if |dx|<1 and |dy|<1
    from scipy.optimize import milp, LinearConstraint, Bounds
    import scipy.sparse as sp
    m = len(cand)
    dx = np.abs(cand[:, None, 0] - cand[None, :, 0]); dy = np.abs(cand[:, None, 1] - cand[None, :, 1])
    I, J = np.nonzero(np.triu((dx < 1 - 1e-9) & (dy < 1 - 1e-9), 1))
    if len(I):
        A_ = sp.csr_matrix((np.ones(2 * len(I)), (np.r_[np.arange(len(I)), np.arange(len(I))], np.r_[I, J])), shape=(len(I), m))
        cons = [LinearConstraint(A_, -np.inf, 1)]
    else:
        cons = []
    res = milp(-np.ones(m), constraints=cons, integrality=np.ones(m), bounds=Bounds(0, 1), options=dict(time_limit=20))
    pick = cand[res.x > 0.5] if res.x is not None else cand[:0]
    A = list(P) + list(pick)
    return np.array(A), len(P)

def trial(args):
    w, h, seed, f0 = args
    rng = np.random.default_rng(seed)
    theta = rng.uniform(math.radians(12), math.radians(40))
    c, s = math.cos(theta), math.sin(theta)
    # lattice patch spanning a random sub-box of the gadget
    bw = rng.uniform(0.5, 1.0) * w; bh = rng.uniform(0.5, 1.0) * h
    cx = rng.uniform(bw / 2, w + f0 - bw / 2); cy = rng.uniform(bh / 2, h + f0 - bh / 2)
    ox, oy = cx + rng.uniform(-0.5, 0.5), cy + rng.uniform(-0.5, 0.5)
    R = int(max(w, h)) + 2
    # keep lattice points whose (i, j) lie in a parallelogram of the requested size
    ni = int(bw * c + bh * s) // 2 + 1; nj = int(bh * c + bw * s) // 2 + 1
    A, nt = build(w, h, f0, theta, ox, oy, -ni, ni, -nj, nj, rng)
    n = len(A); d = n - w * h
    if d < 1 or nt < 4:
        return dict(w=w, h=h, seed=seed, theta=math.degrees(theta), n=n, d=d, f=None)
    t0 = time.time()
    x, y, t, L, vio, L0 = slp(A[:, 0], A[:, 1], A[:, 2], iters=400, ex=w, ey=h)
    ok = vio < 1e-10
    return dict(w=w, h=h, seed=seed, theta=math.degrees(theta), n=n, d=d, nt=nt, f0=L0, f=float(L) if ok else None,
                v=float(vio), dt=time.time() - t0, xyt=np.c_[x, y, t].tolist() if ok else None)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('w', type=int); ap.add_argument('h', type=int)
    ap.add_argument('--trials', type=int, default=50); ap.add_argument('--procs', type=int, default=4)
    ap.add_argument('--seed0', type=int, default=0); ap.add_argument('--f0', type=float, default=0.95)
    ap.add_argument('--out', default='gen.jsonl')
    a = ap.parse_args()
    jobs = [(a.w, a.h, a.seed0 + k, a.f0) for k in range(a.trials)]
    with Pool(a.procs) as pool, open(a.out, 'a') as fo:
        for r in pool.imap_unordered(trial, jobs):
            fo.write(json.dumps(r) + '\n'); fo.flush()
            if r['f'] is not None:
                print(f"{r['w']}x{r['h']}+{r['d']} theta {r['theta']:.1f} f {r['f']:.6f} (nt {r['nt']}, {r['dt']:.0f}s)", flush=True)
