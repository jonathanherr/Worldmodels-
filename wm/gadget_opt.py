"""Shrink fault-line gadgets: pack m unit squares into the smallest (w + f) x (h + f) rectangle.

A gadget for wm.fault_dp is exactly a rectangle packing: m = w*h + d squares inside
[0, w + f] x [0, h + f]. Every n whose best chain uses (w, h, d) gains when its f drops, so a
better gadget pays off across many puzzles at once. This reuses the overlap energy of
wm.pack_opt (pairs and the lower walls, with the upper walls pushed out of the way) and adds the
two upper walls at w + f and h + f, with f a variable:

    compact: minimize f + mu * E(x, y, t, f) with L-BFGS for growing mu
    squeeze: then lower f while relaxing at fixed f stays exactly feasible (with a margin)
"""

import numpy as np
from scipy.optimize import minimize

from .pack_opt import MARGIN, energy_grad, relax_fire

BIG = 1e6


def _walls(x, y, t, W, H):
    """Upper-wall penetration energy and gradients for the box [0, W] x [0, H]."""
    hw = 0.5 * (np.abs(np.cos(t)) + np.abs(np.sin(t)))
    dhw = 0.5 * (-np.sign(np.cos(t)) * np.sin(t) + np.sign(np.sin(t)) * np.cos(t))
    px = np.maximum(x + hw - W, 0)
    py = np.maximum(y + hw - H, 0)
    e = (px ** 2).sum() + (py ** 2).sum()
    gx, gy = 2 * px, 2 * py
    gt = 2 * (px + py) * dhw
    gf = -2 * (px.sum() + py.sum())
    return e, gx, gy, gt, gf


def energy(x, y, t, w, h, f, margin=MARGIN):
    """Energy and gradient (gx, gy, gt, gf) of a gadget in [0, w + f] x [0, h + f]; worst pair depth."""
    e, gx, gy, gt, _, worst = energy_grad(x, y, t, BIG, margin)
    ew, wx, wy, wt, gf = _walls(x, y, t, w + f, h + f)
    return e + ew, gx + wx, gy + wy, gt + wt, gf, worst, ew


def bloat(x, y, t, w, h):
    hw = 0.5 * (np.abs(np.cos(t)) + np.abs(np.sin(t)))
    return max(0.0, (x + hw).max() - w, (y + hw).max() - h)


def feasible(x, y, t, w, h, f, margin=MARGIN):
    *_, worst, ew = energy(x, y, t, w, h, f, margin)
    hw = 0.5 * (np.abs(np.cos(t)) + np.abs(np.sin(t)))
    return worst <= 0 and np.all(x - hw >= -1e-12) and np.all(y - hw >= -1e-12) and bloat(x, y, t, w, h) <= f + 1e-12


def compact(x, y, t, w, h, f, mus=(10.0, 100.0, 1e3, 1e4, 1e5), maxiter=400, margin=MARGIN):
    n = len(x)
    z = np.concatenate([x, y, t, [f]]).astype(np.float64)
    for mu in mus:
        def obj(z):
            e, gx, gy, gt, gf, _, _ = energy(z[:n], z[n:2 * n], z[2 * n:3 * n], w, h, z[-1], margin)
            return z[-1] + mu * e, np.concatenate([mu * gx, mu * gy, mu * gt, [1.0 + mu * gf]])

        z = minimize(obj, z, jac=True, method="L-BFGS-B", options={"maxiter": maxiter, "maxcor": 20}).x
    return z[:n], z[n:2 * n], z[2 * n:3 * n], float(z[-1])


def relax(x, y, t, w, h, f, margin=MARGIN, maxiter=1500):
    """Minimize the energy at fixed f (positions and angles only)."""
    n = len(x)
    z = np.concatenate([x, y, t]).astype(np.float64)

    def obj(z):
        e, gx, gy, gt, *_ = energy(z[:n], z[n:2 * n], z[2 * n:], w, h, f, margin)
        return e, np.concatenate([gx, gy, gt])

    z = minimize(obj, z, jac=True, method="L-BFGS-B", options={"maxiter": maxiter, "maxcor": 20, "ftol": 1e-30, "gtol": 1e-14}).x
    return z[:n], z[n:2 * n], z[2 * n:]


def _stretch(x, y, w, h, f_from, f_to):
    return x * (w + f_to) / (w + f_from), y * (h + f_to) / (h + f_from)


def squeeze(x, y, t, w, h, f, rounds=30, step=0.01, min_step=1e-7, margin=MARGIN):
    """Lower f while the gadget relaxes to an exactly feasible one. Returns (x, y, t, f) or None."""
    for _ in range(30):
        x, y, t = relax(x, y, t, w, h, f, margin)
        if feasible(x, y, t, w, h, f, margin):
            break
        x, y = _stretch(x, y, w, h, f, f + 0.005)
        f += 0.005
    else:
        return None
    best = (x, y, t, f)
    for _ in range(rounds):
        if step < min_step:
            break
        bx, by, bt, bf = best
        ft = bf - step
        sx, sy = _stretch(bx, by, w, h, bf, ft)
        nx, ny, nt = relax(sx, sy, bt, w, h, ft, margin)
        if feasible(nx, ny, nt, w, h, ft, margin):
            best = (nx, ny, nt, ft)
            step *= 1.5
        else:
            step *= 0.4
    return best


def improve(sq, w, h, rng, kicks=8, sigma=0.05, margin=MARGIN):
    """Basin hopping from a gadget's squares (x, y, angle rows): returns (squares, f) of the best."""
    x, y, t = sq[:, 0].copy(), sq[:, 1].copy(), sq[:, 2].copy()
    f0 = bloat(x, y, t, w, h)
    best = (sq.copy(), f0)
    cur = (x, y, t, f0)
    for k in range(kicks + 1):
        x, y, t, f = cur
        if k:
            x = x + rng.normal(0, sigma, len(x))
            y = y + rng.normal(0, sigma, len(y))
            t = t + rng.normal(0, sigma, len(t))
        cx, cy, ct, cf = compact(x, y, t, w, h, f)
        res = squeeze(cx, cy, ct, w, h, max(cf, 0.0) + 0.002, margin=margin)
        if res is None:
            continue
        rx, ry, rt, rf = res
        rf = bloat(rx, ry, rt, w, h)
        if rf < best[1] - 1e-10:
            best = (np.stack([rx, ry, rt], 1), rf)
            cur = (rx, ry, rt, rf)
    return best
