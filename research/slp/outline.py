"""Tilted-block gadget search over block outlines.

A gadget w x h at box (w+f0) x (h+f0) = a patch of a square lattice tilted by theta with origin o,
given row by row (row j of the lattice holds squares i = a_j..b_j), plus an exact (MILP) maximum
set of axis squares from four corner-anchored integer grids. Hill-climb the outline, origin and
angle to maximise the count at f0, then SLP-polish the best to minimise f with the count fixed.

    python research/slp/outline.py W H --f0 0.80 --seed-npy gad.npy --steps 2000 --restarts 4
"""
import argparse, json, math, os, sys, time
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import numpy as np
import scipy.sparse as sp
from scipy.optimize import milp, LinearConstraint, Bounds
import multiprocessing as mp

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from slp import slp, corners


def tilted_squares(theta, o, rows):
    u = np.array([math.cos(theta), math.sin(theta)]); v = np.array([-math.sin(theta), math.cos(theta)])
    pts = [o + (i + s) * u + j * v for j, (a, b, s) in rows.items() for i in range(a, b + 1)]
    P = np.array(pts).reshape(-1, 2)
    return np.c_[P, np.full(len(P), theta)]


def in_box(S, W, H, eps=1e-9):
    cx, cy, _, _ = corners(S[:, 0], S[:, 1], S[:, 2])
    return (cx.min(1) >= -eps) & (cy.min(1) >= -eps) & (cx.max(1) <= W + eps) & (cy.max(1) <= H + eps)


def axis_conflicts_tilted(C, T):
    """bool per axis candidate: overlaps any tilted square (SAT, vectorised over pairs within 1.42)."""
    if len(T) == 0:
        return np.zeros(len(C), bool)
    d = np.hypot(C[:, None, 0] - T[None, :, 0], C[:, None, 1] - T[None, :, 1])
    I, J = np.nonzero(d < 1.4143)
    bad = np.zeros(len(C), bool)
    if len(I) == 0:
        return bad
    tx, ty, _, _ = corners(T[J, 0], T[J, 1], T[J, 2])  # (p, 4)
    ax0, ax1 = C[I, 0] - 0.5, C[I, 0] + 0.5
    ay0, ay1 = C[I, 1] - 0.5, C[I, 1] + 0.5
    tol = 1e-9
    sep = (tx.max(1) <= ax0 + tol) | (tx.min(1) >= ax1 - tol) | (ty.max(1) <= ay0 + tol) | (ty.min(1) >= ay1 - tol)
    # tilted square's own axes
    th = T[J, 2]
    for ang in (th, th + math.pi / 2):
        c, s = np.cos(ang), np.sin(ang)
        axc = np.stack([ax0, ax1, ax1, ax0], 1); ayc = np.stack([ay0, ay0, ay1, ay1], 1)
        pa = axc * c[:, None] + ayc * s[:, None]
        pt = tx * c[:, None] + ty * s[:, None]
        sep |= (pa.max(1) <= pt.min(1) + tol) | (pt.max(1) <= pa.min(1) + tol)
    np.logical_or.at(bad, I, ~sep)
    return bad


def frame_fill(T, w, h, W, H, time_limit=0.5):
    cand = []
    for gx in range(w + 1):
        for gy in range(h + 1):
            for x in (gx + 0.5, W - gx - 0.5):
                for y in (gy + 0.5, H - gy - 0.5):
                    cand.append((x, y))
    C = np.unique(np.round(np.array(cand), 9), axis=0)
    C = C[(C[:, 0] >= 0.5 - 1e-9) & (C[:, 0] <= W - 0.5 + 1e-9) & (C[:, 1] >= 0.5 - 1e-9) & (C[:, 1] <= H - 0.5 + 1e-9)]
    C = C[~axis_conflicts_tilted(C, T)]
    m = len(C)
    if m == 0:
        return C
    dx = np.abs(C[:, None, 0] - C[None, :, 0]); dy = np.abs(C[:, None, 1] - C[None, :, 1])
    I, J = np.nonzero(np.triu((dx < 1 - 1e-9) & (dy < 1 - 1e-9), 1))
    cons = []
    if len(I):
        A = sp.csr_matrix((np.ones(2 * len(I)), (np.r_[np.arange(len(I)), np.arange(len(I))], np.r_[I, J])), shape=(len(I), m))
        cons = [LinearConstraint(A, -np.inf, 1)]
    res = milp(-np.ones(m), constraints=cons, integrality=np.ones(m), bounds=Bounds(0, 1), options=dict(time_limit=time_limit))
    if res.x is None:
        return C[:0]
    return C[res.x > 0.5]


def evaluate(state, w, h, f0):
    theta, o, rows = state
    W, H = w + f0, h + f0
    T = tilted_squares(theta, o, rows)
    if len(T) == 0 or not in_box(T, W, H).all():
        return -1, None
    A = frame_fill(T, w, h, W, H)
    S = np.vstack([T, np.c_[A, np.zeros(len(A))]])
    return len(S), S


def _fit_strips(P, theta, tol=0.1):
    u = np.array([math.cos(theta), math.sin(theta)]); v = np.array([-math.sin(theta), math.cos(theta)])
    B = (P @ v)[:, None] - (P @ v)[None, :]
    score = (np.abs(B - np.round(B)) < tol).sum(0)
    good = score == score.max()
    o = P[good][np.argmin((P[good] @ v))].astype(float)
    P = P[np.abs((P - o) @ v - np.round((P - o) @ v)) < tol]
    a_, b_ = (P - o) @ u, (P - o) @ v
    jj = np.round(b_).astype(int)
    rows = {}; kept = 0
    for j in sorted(set(jj)):
        x = a_[jj == j]
        s = float(np.median(np.mod(x, 1.0)))
        ok = np.abs(np.mod(x - s + 0.5, 1.0) - 0.5) < tol
        ii = np.round(x[ok] - s).astype(int)
        if len(ii):
            rows[int(j)] = (int(ii.min()), int(ii.max()), s); kept += int(ii.max() - ii.min() + 1)
    return (theta, o, rows), kept


def seed_from(q, w, h):
    """Fit strips (rows along u or along v, each with its own slide offset) to a gadget's tilted squares."""
    ang = np.mod(q[:, 2], math.pi / 2)
    tilt = (ang > 0.05) & (ang < math.pi / 2 - 0.05)
    theta = float(np.median(ang[tilt]))
    P = q[tilt & (np.abs(ang - theta) < math.radians(2.0)), :2]
    fits = [_fit_strips(P, th) for th in (theta, theta + math.pi / 2)]
    return max(fits, key=lambda f: f[1])[0]


def clip_start(theta, w, h, f0, mx, my, slides, o=None):
    """Lattice strips clipped to the inner box [mx, W-mx] x [my, H-my]; slides(j) gives row j's offset."""
    W, H = w + f0, h + f0
    u = np.array([math.cos(theta), math.sin(theta)]); v = np.array([-math.sin(theta), math.cos(theta)])
    if o is None:
        o = np.array([W / 2, H / 2])
    R = int(W + H) + 2
    rows = {}
    for j in range(-R, R + 1):
        s = slides(j)
        ii = np.arange(-R, R + 1)
        P = o[None, :] + (ii[:, None] + s) * u[None, :] + j * v[None, :]
        T = np.c_[P, np.full(len(P), theta)]
        cx, cy, _, _ = corners(T[:, 0], T[:, 1], T[:, 2])
        ok = (cx.min(1) >= mx) & (cy.min(1) >= my) & (cx.max(1) <= W - mx) & (cy.max(1) <= H - my)
        if ok.any():
            k = ii[ok]
            rows[j] = (int(k.min()), int(k.max()), float(s))
            # keep contiguous only
            if len(k) != k.max() - k.min() + 1:
                rows[j] = (int(k.min()), int(k.min()), float(s))
    return theta, o.astype(float), rows


def random_start(w, h, f0, rng):
    theta = math.radians(rng.uniform(18, 38)) + (math.pi / 2 if rng.random() < 0.5 else 0.0)
    mode = rng.integers(3)
    ph = rng.random()
    if mode == 0:
        slides = lambda j: 0.0
    elif mode == 1:
        slides = lambda j: (0.5 * (j % 2) + ph) % 1.0
    else:
        st = rng.random()
        slides = lambda j: (st * j + ph) % 1.0
    mx = rng.uniform(0.3, 2.5); my = rng.uniform(0.3, 2.5)
    o = np.array([rng.uniform(0, w + f0), rng.uniform(0, h + f0)])
    return clip_start(theta, w, h, f0, mx, my, slides, o)


def neighbour(state, rng):
    theta, o, rows = state
    rows = dict(rows); o = o.copy()
    r = rng.random()
    js = sorted(rows)
    if r < 0.6:
        j = js[rng.integers(len(js))]; a, b, s = rows[j]
        if rng.random() < 0.5: a += rng.choice([-1, 1])
        else: b += rng.choice([-1, 1])
        if a <= b: rows[j] = (a, b, s)
        elif len(rows) > 1: del rows[j]
    elif r < 0.7:
        j = (js[0] - 1) if rng.random() < 0.5 else (js[-1] + 1)
        nb = rows[js[0]] if j < js[0] else rows[js[-1]]
        rows[j] = nb
    elif r < 0.8:
        o += rng.normal(0, 0.03, 2)
    elif r < 0.9:
        j = js[rng.integers(len(js))]; a, b, s = rows[j]
        rows[j] = (a, b, s + rng.normal(0, 0.05))
    else:
        theta += rng.normal(0, math.radians(0.15))
    return theta, o, rows


def climb(args, verbose=False):
    w, h, f0, seed, steps, start = args
    rng = np.random.default_rng(seed)
    state = start
    if state is None:
        best = None
        for _ in range(40):
            st = random_start(w, h, f0, rng)
            if not st[2]:
                continue
            n_, _ = evaluate(st, w, h, f0)
            if best is None or n_ > best[0]:
                best = (n_, st)
        state = best[1]
    best_n, best_S = evaluate(state, w, h, f0)
    cur_n = best_n
    t0 = time.time()
    for k in range(steps):
        if verbose and k % 500 == 0:
            print(f'  seed {seed} step {k} best {best_n} ({time.time() - t0:.0f}s)', flush=True)
        cand = neighbour(state, rng)
        n, S = evaluate(cand, w, h, f0)
        if n >= cur_n:
            state, cur_n = cand, n
            if n > best_n:
                best_n, best_S = n, S
    return best_n, best_S, state


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('w', type=int); ap.add_argument('h', type=int)
    ap.add_argument('--f0', type=float, default=0.80)
    ap.add_argument('--seed-npy'); ap.add_argument('--theta', type=float, default=26.57)
    ap.add_argument('--steps', type=int, default=1000); ap.add_argument('--restarts', type=int, default=4)
    ap.add_argument('--procs', type=int, default=4); ap.add_argument('--out', default='outline')
    a = ap.parse_args()
    w, h = a.w, a.h
    if a.seed_npy:
        start = seed_from(np.load(a.seed_npy), w, h)
    else:
        start = None
    if start is not None:
        n0, _ = evaluate(start, w, h, a.f0)
        print(f'start count {n0} (need > {w * h})', flush=True)
    t0 = time.time()
    with mp.get_context('spawn').Pool(a.procs) as pool:
        res = pool.map(climb, [(w, h, a.f0, s, a.steps, start) for s in range(a.restarts)])
    res.sort(key=lambda r: -r[0])
    for n, S, st in res:
        print(f'climb: count {n} = {w}x{h}+{n - w * h} at f0 {a.f0} ({time.time() - t0:.0f}s)', flush=True)
    n, S, st = res[0]
    if S is not None and n > w * h:
        x, y, t, L, v, L0 = slp(S[:, 0], S[:, 1], S[:, 2], iters=400, ex=w, ey=h)
        print(f'SLP: {w}x{h}+{n - w * h} f {L0:.6f} -> {L:.6f} viol {v:.1e}', flush=True)
        if v < 1e-10:
            np.save(f'{a.out}_{w}_{h}_{n - w * h}.npy', np.c_[x, y, t])
