"""Grow-and-cut sticks: re-pack a packing as rows of squares whose lengths are free, then cut.

Large-n record packings are bands of tilted "sticks" (straight runs of touching squares at one
angle) around an upright core, which is itself a set of axis-aligned sticks. Here a packing is
split into sticks, each stick becomes a 1 x l rectangle with a continuous length l, and in a box
slightly smaller than the record the sticks are pushed to grow (maximize total length) while their
separating-axis overlap is penalized. The neighbours stop each stick where it can grow no further,
so they act as the scissors: lengths are cut down to whole squares, the sticks whose fractional
part is large are offered one more square if the relaxation can absorb it, and if the sticks hold
at least n squares the packing goes to `pack_opt.squeeze` and the official-metric store.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from scipy.optimize import minimize

from .pack_opt import MARGIN, feasible, squeeze

QUARTER = math.pi / 2


def _norm(t):
    """Angle modulo 90 degrees, in [-45, 45) degrees (a square looks the same under 90-degree turns)."""
    return (np.asarray(t) + math.pi / 4) % QUARTER - math.pi / 4


def split_sticks(x, y, t, ang_tol=1e-3, pos_tol=0.03, gap_max=0.06):
    """Split a packing into sticks. Returns a list of (indices along the stick, stick angle phi).

    The stick's long axis is (-sin phi, cos phi). Each square joins the chain direction (its local
    e2 or e1 axis) in which its run of touching, same-angle neighbours is longer.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    th = _norm(t)
    n = len(x)
    nxt = [dict(), dict()]  # per direction: i -> j where j sits next to i along +axis
    for i in range(n):
        d = np.hypot(x - x[i], y - y[i])
        for j in np.nonzero((d > 0.5) & (d < 1 + gap_max + pos_tol))[0]:
            if abs(_norm(th[j] - th[i])) > ang_tol:
                continue
            c, s = math.cos(th[i]), math.sin(th[i])
            dx, dy = x[j] - x[i], y[j] - y[i]
            for k, (ax, ay, px, py) in enumerate(((-s, c, c, s), (c, s, -s, c))):
                along, perp = dx * ax + dy * ay, dx * px + dy * py
                if abs(perp) < pos_tol and 1 - pos_tol < along < 1 + gap_max:
                    nxt[k][i] = j
    runs = []
    for k in range(2):
        prv = {j: i for i, j in nxt[k].items()}
        length = np.zeros(n, int)
        for i in range(n):
            if i in prv:
                continue
            chain = [i]
            while chain[-1] in nxt[k] and len(chain) <= n:
                chain.append(nxt[k][chain[-1]])
            length[chain] = len(chain)
        runs.append(length)
    use = (runs[1] > runs[0]).astype(int)  # 0: stick along e2, 1: along e1
    sticks, seen = [], np.zeros(n, bool)
    for i in range(n):
        k = use[i]
        p = i
        while True:  # walk back to the start of i's chain among squares that chose direction k
            q = next((a for a, b in nxt[k].items() if b == p and use[a] == k and not seen[a]), None)
            if q is None or q == i:
                break
            p = q
        if seen[p]:
            continue
        chain = [p]
        seen[p] = True
        while chain[-1] in nxt[k] and use[nxt[k][chain[-1]]] == k and not seen[nxt[k][chain[-1]]]:
            chain.append(nxt[k][chain[-1]])
            seen[chain[-1]] = True
        phi = float(th[p]) - k * QUARTER
        sticks.append((chain, phi))
    for i in np.nonzero(~seen)[0]:
        sticks.append(([int(i)], float(th[i])))
    return sticks


def stick_params(x, y, sticks):
    """(cx, cy, phi, length) arrays of the sticks, length = number of squares."""
    cx = np.array([np.mean(np.asarray(x)[c]) for c, _ in sticks])
    cy = np.array([np.mean(np.asarray(y)[c]) for c, _ in sticks])
    phi = np.array([p for _, p in sticks])
    ln = np.array([float(len(c)) for c, _ in sticks])
    return cx, cy, phi, ln


def rect_energy(cx, cy, phi, ln, L, I, J, margin=MARGIN):
    """Overlap energy of 1 x ln rectangles (torch, differentiable) plus wall penetration."""
    a = torch.stack([phi.cos(), phi.sin()], 1)  # across, half-width 1/2
    b = torch.stack([-phi.sin(), phi.cos()], 1)  # along, half-length ln/2
    d = torch.stack([cx[J] - cx[I], cy[J] - cy[I]], 1)
    best = None
    for u in (a[I], b[I], a[J], b[J]):
        r = (0.5 * (u * a[I]).sum(1).abs() + 0.5 * ln[I] * (u * b[I]).sum(1).abs()
             + 0.5 * (u * a[J]).sum(1).abs() + 0.5 * ln[J] * (u * b[J]).sum(1).abs())
        dep = r + margin - (d * u).sum(1).abs()
        best = dep if best is None else torch.minimum(best, dep)
    e = best.clamp(min=0).pow(2).sum()
    hx = 0.5 * phi.cos().abs() + 0.5 * ln * phi.sin().abs()
    hy = 0.5 * phi.sin().abs() + 0.5 * ln * phi.cos().abs()
    for lo, hi in ((cx - hx, cx + hx), (cy - hy, cy + hy)):
        e = e + (-lo).clamp(min=0).pow(2).sum() + (hi - L).clamp(min=0).pow(2).sum()
    return e


def _pairs(m):
    I, J = np.triu_indices(m, 1)
    return torch.as_tensor(I), torch.as_tensor(J)


def _solve(z0, L, m, I, J, grow, mus, fixed_len=None, maxiter=400):
    """Minimize -grow * sum(len) + mu * E over (cx, cy, phi[, len]) for increasing mu."""
    z = z0.copy()
    for mu in mus:
        def f(zz):
            v = torch.tensor(zz, dtype=torch.float64, requires_grad=True)
            cx, cy, phi = v[:m], v[m:2 * m], v[2 * m:3 * m]
            ln = torch.as_tensor(fixed_len) if fixed_len is not None else v[3 * m:]
            e = rect_energy(cx, cy, phi, ln, L, I, J)
            obj = mu * e - (grow * ln.sum() if fixed_len is None else 0.0)
            obj.backward()
            return float(obj.detach()), v.grad.numpy().copy()

        bounds = None
        if fixed_len is None:
            bounds = [(None, None)] * (3 * m) + [(1.0, None)] * m
        z = minimize(f, z, jac=True, method="L-BFGS-B", bounds=bounds,
                     options={"maxiter": maxiter, "maxcor": 20}).x
    return z


def energy_at(z, L, m, I, J, ln):
    v = torch.tensor(z[:3 * m], dtype=torch.float64)
    return float(rect_energy(v[:m], v[m:2 * m], v[2 * m:], torch.as_tensor(ln, dtype=torch.float64), L, I, J))


def to_squares(cx, cy, phi, ln):
    """Unit squares laid along each stick: ln[i] (an integer) squares centred on the stick."""
    xs, ys, ts = [], [], []
    for c_x, c_y, p, k in zip(cx, cy, phi, ln.astype(int)):
        bx, by = -math.sin(p), math.cos(p)
        for q in range(k):
            o = q - (k - 1) / 2
            xs.append(c_x + o * bx)
            ys.append(c_y + o * by)
            ts.append(p)
    return np.array(xs), np.array(ys), np.array(ts)


def grow_and_cut(x, y, t, L0, n, eps, grow=1.0, tol=1e-12, log=None):
    """One attempt at side L0 * (1 - eps). Returns a feasible (x, y, t, L) of >= n squares, or None."""
    sticks = split_sticks(x, y, t)
    cx, cy, phi, ln = stick_params(x, y, sticks)
    m = len(sticks)
    I, J = _pairs(m)
    L = L0 * (1 - eps)
    k = L / L0
    z = np.concatenate([cx * k, cy * k, phi, ln * k])
    # Grow: lengths free, overlap penalized harder and harder.
    z = _solve(z, L, m, I, J, grow, mus=(1e2, 1e3, 1e4, 1e5))
    grown = z[3 * m:]
    cut = np.floor(grown + 1e-6)
    if log:
        log(f"    eps={eps:.0e} sticks={m} grown={grown.sum():.2f} cut={int(cut.sum())} need={n}")
    if cut.sum() < n:
        return None
    # Scissors: settle at the cut lengths, then try one extra square on the sticks that grew most.
    z = _solve(z, L, m, I, J, 0.0, mus=(1e4, 1e6), fixed_len=cut)
    extra = n - cut.sum()
    if extra < 0:  # more squares than needed: drop from the sticks that were cut back hardest
        for i in np.argsort(grown - cut)[::-1]:
            while extra < 0 and cut[i] > 1:
                cut[i] -= 1
                extra += 1
    sx, sy, st = to_squares(z[:m], z[m:2 * m], z[2 * m:3 * m], cut)
    if len(sx) < n:
        return None
    sx, sy, st = sx[:n], sy[:n], st[:n]
    out = squeeze(sx, sy, st, L, rounds=25, step=max(eps, 1e-5), maxiter=2000, method="lbfgs")
    if out is None or not feasible(*out[:3], out[3], MARGIN):
        return None
    return out
