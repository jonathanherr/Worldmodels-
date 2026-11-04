"""Sequential linear programming (SLP) polish for unit-square packings.

Every step solves one LP: minimise the box side L over small moves (dx, dy, dtheta) of every square,
subject to linearised non-overlap (corners of one square stay outside a face half-plane of the
other, the face chosen as the best separating axis at the current point) and linearised walls.
A trust region keeps the linearisation honest; a merit test (L + mu * violation) accepts steps.

Unlike the penalty squeeze in wm/pack_opt.py this converges to a KKT point of the exact problem
(all binding contacts move together), which is what "converged" means for Kaggle polish.
"""

from __future__ import annotations

import math

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog
from scipy.spatial import cKDTree

V = np.array([[0.5, 0.5], [-0.5, 0.5], [-0.5, -0.5], [0.5, -0.5]])


def corners(x, y, t):
    c, s = np.cos(t), np.sin(t)
    # (n, 4, 2)
    rx = c[:, None] * V[None, :, 0] - s[:, None] * V[None, :, 1]
    ry = s[:, None] * V[None, :, 0] + c[:, None] * V[None, :, 1]
    return x[:, None] + rx, y[:, None] + ry, rx, ry


def side(x, y, t, ex=0.0, ey=0.0):
    cx, cy, _, _ = corners(x, y, t)
    return max(cx.max() - cx.min() - ex, cy.max() - cy.min() - ey), cx.min(), cy.min()


def pairs_of(x, y, r=1.75):
    tree = cKDTree(np.c_[x, y])
    return np.array(sorted(tree.query_pairs(r)), dtype=np.int64).reshape(-1, 2)


def _sep_rows(x, y, t, P, cut=0.3):
    """For each pair pick the best separating face; return linear rows (f0, idx, coef)."""
    cx, cy, rx, ry = corners(x, y, t)
    rows = []
    minsep = np.inf
    for i, j in P:
        best = None
        for a, b in ((i, j), (j, i)):
            for k in range(4):
                phi = t[a] + k * math.pi / 2
                u = np.array([math.cos(phi), math.sin(phi)])
                f = u[0] * (cx[b] - x[a]) + u[1] * (cy[b] - y[a]) - 0.5
                m = f.min()
                if best is None or m > best[0]:
                    best = (m, a, b, phi, f)
        m, a, b, phi, f = best
        minsep = min(minsep, m)
        if m > 0.45:  # far apart even after a full trust step
            continue
        cphi, sphi = math.cos(phi), math.sin(phi)
        for q in range(4):
            if f[q] > m + cut:
                continue
            # f = u.(cB + R v - cA) - .5
            dcx, dcy = cx[b][q] - x[a], cy[b][q] - y[a]
            g_ta = -sphi * dcx + cphi * dcy  # d/dtheta_a (rotating u)
            g_tb = cphi * (-ry[b][q]) + sphi * rx[b][q]  # u . (J R v)
            rows.append((f[q], (3 * a, 3 * a + 1, 3 * a + 2, 3 * b, 3 * b + 1, 3 * b + 2),
                         (-cphi, -sphi, g_ta, cphi, sphi, g_tb)))
    return rows, minsep


def violation(x, y, t, P, L, ex=0.0, ey=0.0):
    _, minsep = _sep_rows(x, y, t, P, cut=-1)
    cx, cy, _, _ = corners(x, y, t)
    wall = max(-cx.min(), -cy.min(), cx.max() - L - ex, cy.max() - L - ey, 0.0)
    return max(0.0, -minsep, wall)


def lp_step(x, y, t, L, P, delta, dtheta, gap=0.0, ex=0.0, ey=0.0):
    n = len(x)
    nv = 3 * n + 1
    rows, _ = _sep_rows(x, y, t, P)
    cx, cy, rx, ry = corners(x, y, t)
    data, ri, ci, rhs = [], [], [], []
    r = 0
    # pair rows: f0 + g.d >= gap  ->  -g.d <= f0 - gap
    for f0, idx, coef in rows:
        for c_, v_ in zip(idx, coef):
            ri.append(r); ci.append(c_); data.append(-v_)
        rhs.append(f0 - gap)
        r += 1
    # walls, only corners near a wall
    near = 0.3
    for i in range(n):
        for q in range(4):
            jx, jy = -ry[i][q], rx[i][q]  # J R v
            if cx[i][q] < near:  # cx + dx + jx dt >= 0 -> -dx - jx dt <= cx
                ri += [r, r]; ci += [3 * i, 3 * i + 2]; data += [-1.0, -jx]; rhs.append(cx[i][q] - gap); r += 1
            if cy[i][q] < near:
                ri += [r, r]; ci += [3 * i + 1, 3 * i + 2]; data += [-1.0, -jy]; rhs.append(cy[i][q] - gap); r += 1
            if cx[i][q] > L + ex - near:  # cx + dx + jx dt <= L + ex + dL
                ri += [r, r, r]; ci += [3 * i, 3 * i + 2, nv - 1]; data += [1.0, jx, -1.0]
                rhs.append(L + ex - cx[i][q] - gap); r += 1
            if cy[i][q] > L + ey - near:
                ri += [r, r, r]; ci += [3 * i + 1, 3 * i + 2, nv - 1]; data += [1.0, jy, -1.0]
                rhs.append(L + ey - cy[i][q] - gap); r += 1
    A = sp.csr_matrix((data, (ri, ci)), shape=(r, nv))
    c = np.zeros(nv); c[-1] = 1.0
    b = np.array([delta, delta, dtheta] * n + [1.0])
    bounds = list(zip(-b, b))
    res = linprog(c, A_ub=A, b_ub=np.array(rhs), bounds=bounds, method="highs")
    if res.status != 0:
        return None
    d = res.x
    return d[0:-1:3], d[1:-1:3], d[2:-1:3], d[-1]


def normalise(x, y, t):
    _, x0, y0 = side(x, y, t)
    return x - x0, y - y0


def slp(x, y, t, iters=300, delta=0.02, min_delta=1e-9, mu=50.0, verbose=False, log_every=20, ex=0.0, ey=0.0, gap=2e-10):
    x, y, t = (np.asarray(a, float).copy() for a in (x, y, t))
    x, y = normalise(x, y, t)
    L, _, _ = side(x, y, t, ex, ey)
    P = pairs_of(x, y)
    v = violation(x, y, t, P, L, ex, ey)
    L0 = L
    dth = delta
    hist = []
    fails = 0
    for it in range(iters):
        if it % 10 == 0:
            P = pairs_of(x, y)
        st = lp_step(x, y, t, L, P, delta, dth, gap=gap, ex=ex, ey=ey)
        if st is None:  # linearised problem infeasible inside the trust region: widen it
            fails += 1
            if fails > 6:
                break
            delta = min(max(delta * 4, 20 * v), 0.05); dth = delta
            continue
        fails = 0
        dx, dy, dt, dL = st
        nx, ny, nt = x + dx, y + dy, t + dt
        nx, ny = normalise(nx, ny, nt)
        nL, _, _ = side(nx, ny, nt, ex, ey)
        nv = violation(nx, ny, nt, P, nL, ex, ey)
        if nL + mu * nv < L + mu * v - 1e-15:
            x, y, t, L, v = nx, ny, nt, nL, nv
            if max(np.abs(dx).max(), np.abs(dy).max(), np.abs(dt).max()) > 0.9 * delta:
                delta = min(delta * 1.5, 0.05)
            hist.append(L)
        else:
            delta /= 2
        dth = delta
        if verbose and it % log_every == 0:
            print(f"  it {it} L {L:.9f} v {v:.2e} delta {delta:.1e}", flush=True)
        if delta < min_delta and v < 1e-12:
            break
        if delta < min_delta:
            delta = 20 * v + 1e-9
        if len(hist) > 30 and v < 1e-12 and hist[-30] - hist[-1] < 1e-11:
            break
    return x, y, t, L, v, L0
