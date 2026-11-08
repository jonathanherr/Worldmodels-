"""Packings as a graph of contact structures.

Node  = a polished (KKT) packing of n unit squares, identified by its contact graph: which squares
        touch (and walls), labelled by the world direction of the contact normal. Two packings with
        the same Weisfeiler-Lehman hash of that labelled graph are treated as the same node.
Edge  = a combinatorial move polishing cannot make by itself:
          flip  - a load-bearing contact (a, b) is forced onto a neighbouring face of a
                  (b goes from "right of a" to "above a"), the packing re-converges around it;
          swap  - forced onto the opposite face (a and b trade places; only for pairs whose
                  tilts differ, otherwise it is a relabelling);
          break - the contact is pushed apart by `gap` on its own face.
        After the forced phase the constraint is released and the packing is polished again.

The forced rows are elastic (slack s >= 0, cost mu * s) so the LP stays feasible while the forced
pair is still on the wrong side. Contact loads come from the LP duals at the converged point.
"""

from __future__ import annotations

import math
import os
import sys

import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "slp"))
import slp as S  # noqa: E402


def load_csv_n(path, n):
    xs, ys, ts = [], [], []
    with open(path) as f:
        next(f)
        for line in f:
            p = line.rstrip().split(",")
            if int(p[1]) == n:
                xs.append(float(p[3])); ys.append(float(p[4])); ts.append(float(p[5]))
    return np.array(xs), np.array(ys), np.array(ts)


def best_faces(x, y, t, P):
    """For each pair: owner a, other b, face k (0..3 of a), separation."""
    cx, cy, _, _ = S.corners(x, y, t)
    I, J = P[:, 0], P[:, 1]
    A = np.stack([I, J], 1)[:, :, None].repeat(4, 2).reshape(len(P), 8)
    B = np.stack([J, I], 1)[:, :, None].repeat(4, 2).reshape(len(P), 8)
    K = np.tile(np.arange(4), 2)[None, :].repeat(len(P), 0)
    phi = t[A] + K * (math.pi / 2)
    c, s_ = np.cos(phi), np.sin(phi)
    f = c[..., None] * (cx[B] - x[A][..., None]) + s_[..., None] * (cy[B] - y[A][..., None]) - 0.5
    m = f.min(2)
    best = m.argmax(1)
    r = np.arange(len(P))
    return A[r, best], B[r, best], K[r, best], m[r, best]


def forced_rows(x, y, t, a, b, k):
    """Linear rows for 'every corner of b is outside face k of a': f0 + g.d >= 0."""
    cx, cy, rx, ry = S.corners(x, y, t)
    ph = t[a] + k * math.pi / 2
    cphi, sphi = math.cos(ph), math.sin(ph)
    out = []
    for q in range(4):
        dcx, dcy = cx[b, q] - x[a], cy[b, q] - y[a]
        f0 = cphi * dcx + sphi * dcy - 0.5
        g_ta = -sphi * dcx + cphi * dcy
        g_tb = cphi * (-ry[b, q]) + sphi * rx[b, q]
        idx = [3 * a, 3 * a + 1, 3 * a + 2, 3 * b, 3 * b + 1, 3 * b + 2]
        coef = [-cphi, -sphi, g_ta, cphi, sphi, g_tb]
        out.append((f0, idx, coef))
    return out


def lp_step(x, y, t, L, P, delta, dtheta, force=None, fgap=0.0, mu=30.0, gap=2e-10, want_duals=False):
    """S.lp_step plus elastic forced rows; force = (a, b, k) or None. P must exclude the forced pair."""
    n = len(x)
    nv = 3 * n + 1 + (1 if force else 0)
    iL = 3 * n
    rows, _ = S._sep_rows(x, y, t, P)
    cx, cy, rx, ry = S.corners(x, y, t)
    data, ri, ci, rhs = [], [], [], []
    r = 0
    npair = 0
    if len(rows):
        f0, idx, coef = rows
        m = len(f0)
        ri = list(np.repeat(np.arange(m), 6)); ci = list(idx.ravel()); data = list(-coef.ravel())
        rhs = list(f0 - gap); r = m; npair = m
    near = 0.3
    for i in range(n):
        for q in range(4):
            jx, jy = -ry[i][q], rx[i][q]
            if cx[i][q] < near:
                ri += [r, r]; ci += [3 * i, 3 * i + 2]; data += [-1.0, -jx]; rhs.append(cx[i][q] - gap); r += 1
            if cy[i][q] < near:
                ri += [r, r]; ci += [3 * i + 1, 3 * i + 2]; data += [-1.0, -jy]; rhs.append(cy[i][q] - gap); r += 1
            if cx[i][q] > L - near:
                ri += [r, r, r]; ci += [3 * i, 3 * i + 2, iL]; data += [1.0, jx, -1.0]
                rhs.append(L - cx[i][q] - gap); r += 1
            if cy[i][q] > L - near:
                ri += [r, r, r]; ci += [3 * i + 1, 3 * i + 2, iL]; data += [1.0, jy, -1.0]
                rhs.append(L - cy[i][q] - gap); r += 1
    if force:
        a, b, k = force
        for f0, idx, coef in forced_rows(x, y, t, a, b, k):
            # f0 + g.d + s >= fgap  ->  -g.d - s <= f0 - fgap
            ri += [r] * 7; ci += idx + [nv - 1]; data += [-c for c in coef] + [-1.0]
            rhs.append(f0 - fgap); r += 1
    A = sp.csr_matrix((data, (ri, ci)), shape=(r, nv))
    c = np.zeros(nv); c[iL] = 1.0
    bounds = [(-delta, delta), (-delta, delta), (-dtheta, dtheta)] * n + [(-1.0, 1.0)]
    if force:
        c[-1] = mu
        bounds.append((0.0, None))
    res = linprog(c, A_ub=A, b_ub=np.array(rhs), bounds=bounds, method="highs")
    if res.status != 0:
        return None
    d = res.x
    out = (d[0:3 * n:3], d[1:3 * n:3], d[2:3 * n:3], d[iL], d[-1] if force else 0.0)
    if want_duals:
        lam = -res.ineqlin.marginals[:npair] if npair else np.zeros(0)
        return out, rows, lam
    return out


def forced_violation(x, y, t, force, fgap):
    if not force:
        return 0.0
    a, b, k = force
    return max(0.0, max(fgap - r[0] for r in forced_rows(x, y, t, a, b, k)))


def run_forced(x, y, t, force, fgap=0.0, iters=250, delta=0.03, mu=50.0):
    """SLP with the forced pair constrained to face k (elastic), other pairs on their best face."""
    x, y, t = x.copy(), y.copy(), t.copy()
    a, b, _ = force
    x, y = S.normalise(x, y, t)
    L, _, _ = S.side(x, y, t)

    def pairs():
        P = S.pairs_of(x, y)
        keep = ~(((P[:, 0] == a) & (P[:, 1] == b)) | ((P[:, 0] == b) & (P[:, 1] == a)))
        return P[keep]

    P = pairs()
    merit = lambda X, Y, T, LL, PP: LL + mu * (S.violation(X, Y, T, PP, LL) + forced_violation(X, Y, T, force, fgap))
    cur = merit(x, y, t, L, P)
    for it in range(iters):
        if it % 10 == 0:
            P = pairs(); cur = merit(x, y, t, L, P)
        st = lp_step(x, y, t, L, P, delta, delta, force=force, fgap=fgap, mu=mu)
        if st is None:
            delta = min(delta * 2, 0.08)
            continue
        dx, dy, dt, dL, s = st
        nx_, ny_, nt_ = x + dx, y + dy, t + dt
        nx_, ny_ = S.normalise(nx_, ny_, nt_)
        nL, _, _ = S.side(nx_, ny_, nt_)
        nm = merit(nx_, ny_, nt_, nL, P)
        if nm < cur - 1e-13:
            x, y, t, L, cur = nx_, ny_, nt_, nL, nm
            delta = min(delta * 1.3, 0.08)
        else:
            delta /= 2
        if delta < 1e-7:
            break
    return x, y, t, L, forced_violation(x, y, t, force, fgap)


def contacts(x, y, t, tol=1e-6, duals=True):
    """Contact list with loads. Returns (pairs a,b,k,sep), wall contacts, loads per pair contact."""
    P = S.pairs_of(x, y, r=1.75)
    a, b, k, m = best_faces(x, y, t, P)
    on = m < tol
    ca, cb, ck = a[on], b[on], k[on]
    load = np.zeros(on.sum())
    if duals:
        L, _, _ = S.side(x, y, t)
        res = lp_step(x, y, t, L, P, 1e-4, 1e-4, want_duals=True)
        if res is not None:
            _, rows, lam = res
            f0, idx, coef = rows
            # map each row back to its pair via (owner index, other index)
            ra, rb = idx[:, 0] // 3, idx[:, 3] // 3
            key = {}
            for i, (u, v) in enumerate(zip(ra, rb)):
                key[(min(u, v), max(u, v))] = key.get((min(u, v), max(u, v)), 0.0) + lam[i]
            load = np.array([key.get((min(u, v), max(u, v)), 0.0) for u, v in zip(ca, cb)])
    cx, cy, _, _ = S.corners(x, y, t)
    L, _, _ = S.side(x, y, t)
    walls = []
    for i in range(len(x)):
        if cx[i].min() < tol: walls.append((i, "W"))
        if cy[i].min() < tol: walls.append((i, "S"))
        if cx[i].max() > L - tol: walls.append((i, "E"))
        if cy[i].max() > L - tol: walls.append((i, "N"))
    return ca, cb, ck, load, walls


def graph_of(x, y, t, ca, cb, ck, walls):
    G = nx.Graph()
    for i in range(len(x)):
        tilt = round(((t[i] % (math.pi / 2)) / (math.pi / 2)) * 36) % 36  # 2.5 degree bins
        G.add_node(i, lab=f"t{tilt}")
    for u, v, kk in zip(ca, cb, ck):
        dirn = round(((t[u] + kk * math.pi / 2) % (2 * math.pi)) / (math.pi / 4)) % 8
        G.add_edge(int(u), int(v), lab=f"d{min(dirn, (dirn + 4) % 8)}")
    for side_ in "WSEN":
        G.add_node(side_, lab=side_)
    for i, w in walls:
        G.add_edge(i, w, lab="w")
    return G


def wl_hash(G):
    return nx.weisfeiler_lehman_graph_hash(G, node_attr="lab", edge_attr="lab", iterations=4)


def describe(x, y, t, tol=1e-6):
    ca, cb, ck, load, walls = contacts(x, y, t, tol)
    G = graph_of(x, y, t, ca, cb, ck, walls)
    n = len(x)
    deg = np.array([sum(1 for v in G.neighbors(i)) for i in range(n)])
    rattlers = int((deg <= 2).sum())
    stressed = int((load > 1e-9).sum())
    return dict(n=n, contacts=int(len(ca)), walls=len(walls), dof=3 * n + 1, rattlers=rattlers,
                stressed=stressed, hash=wl_hash(G)), (ca, cb, ck, load, walls)
