"""Soft-shape relaxation for square packings: every square may shrink or stretch, wander, then harden.

The pixel-shader view of packing: each square runs the same small update in parallel, reading only
its neighbours. Rigid polish (`wm.gpu_polish`) and local rules (`wm.life`) keep every square a unit
square, so a jammed record has no free room to move into. Here each square i is a rectangle with
half-sides (a_i, b_i), and a shape penalty

    mu * sum_i (max(0, 1/2 - a_i)^2 + max(0, 1/2 - b_i)^2)

pulls it back to a unit square. With mu small the squares give up size wherever they are crowded,
which opens free volume; noisy parallel steps then let them drift and turn; raising mu hardens them
back to unit squares, and the final packing is checked rigidly at the target side.

Overlap is the separating-axis depth, generalised to rectangles: along an edge normal of the owner
o, depth = r_o + a_q |n . u_q| + b_q |n . v_q| - |d . n|, where r_o is a_o or b_o.
"""

import numpy as np
from numba import njit

from .pack_opt import MARGIN, feasible, relax_lbfgs, squeeze


@njit(cache=True)
def energy_grad(x, y, t, a, b, L, mu, margin):
    """Overlap + wall + shape energy of rectangles in [0, L]^2 and its gradient in (x, y, t, a, b)."""
    n = x.shape[0]
    gx = np.zeros(n)
    gy = np.zeros(n)
    gt = np.zeros(n)
    ga = np.zeros(n)
    gb = np.zeros(n)
    e = 0.0
    c = np.cos(t)
    s = np.sin(t)
    reach = 2.0 + 4 * margin + 1e-9
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[j] - x[i]
            dy = y[j] - y[i]
            if dx * dx + dy * dy >= reach:
                continue
            best = 1e9
            kb = 0
            for k in range(4):
                if k < 2:
                    o, q = i, j
                else:
                    o, q = j, i
                phi = t[q] - t[o]
                cp = abs(np.cos(phi))
                sp = abs(np.sin(phi))
                if k % 2 == 0:
                    nx, ny = c[o], s[o]
                    r = a[o] + a[q] * cp + b[q] * sp
                else:
                    nx, ny = -s[o], c[o]
                    r = b[o] + a[q] * sp + b[q] * cp
                dep = r + margin - abs(dx * nx + dy * ny)
                if dep < best:
                    best = dep
                    kb = k
            if best <= 0.0:
                continue
            e += best * best
            g = 2.0 * best
            if kb < 2:
                o, q = i, j
            else:
                o, q = j, i
            phi = t[q] - t[o]
            cphi = np.cos(phi)
            sphi = np.sin(phi)
            cp = abs(cphi)
            sp = abs(sphi)
            dcp = -np.sign(cphi) * sphi  # d|cos phi| / d phi
            dsp = np.sign(sphi) * cphi
            if kb % 2 == 0:
                nx, ny, mx, my = c[o], s[o], -s[o], c[o]  # m = dn/dt_o
                ga[o] += g
                ga[q] += g * cp
                gb[q] += g * sp
                dr = a[q] * dcp + b[q] * dsp
            else:
                nx, ny, mx, my = -s[o], c[o], -c[o], -s[o]
                gb[o] += g
                ga[q] += g * sp
                gb[q] += g * cp
                dr = a[q] * dsp + b[q] * dcp
            gt[q] += g * dr
            gt[o] -= g * dr
            proj = dx * nx + dy * ny
            sg = 1.0 if proj >= 0 else -1.0
            # depth term -sg * (d . n), d = c_j - c_i
            gx[j] -= g * sg * nx
            gy[j] -= g * sg * ny
            gx[i] += g * sg * nx
            gy[i] += g * sg * ny
            gt[o] -= g * sg * (dx * mx + dy * my)
    for i in range(n):
        ci, si = abs(c[i]), abs(s[i])
        dci = -np.sign(c[i]) * s[i]
        dsi = np.sign(s[i]) * c[i]
        for coord in range(2):
            if coord == 0:
                p = x[i]
                w = a[i] * ci + b[i] * si
                dwt, dwa, dwb = a[i] * dci + b[i] * dsi, ci, si
            else:
                p = y[i]
                w = a[i] * si + b[i] * ci
                dwt, dwa, dwb = a[i] * dsi + b[i] * dci, si, ci
            lo = w + margin - p
            hi = p + w + margin - L
            gp = 0.0
            for v, sgn in ((lo, -1.0), (hi, 1.0)):
                if v > 0:
                    e += v * v
                    gp += 2 * v * sgn
                    gt[i] += 2 * v * dwt
                    ga[i] += 2 * v * dwa
                    gb[i] += 2 * v * dwb
            if coord == 0:
                gx[i] += gp
            else:
                gy[i] += gp
        for k in range(2):
            v = 0.5 - (a[i] if k == 0 else b[i])
            if v > 0:
                e += mu * v * v
                if k == 0:
                    ga[i] -= 2 * mu * v
                else:
                    gb[i] -= 2 * mu * v
    return e, gx, gy, gt, ga, gb


@njit(cache=True)
def run(x, y, t, a, b, L, mus, steps, temp, square, seed, margin):
    """Parallel (Jacobi) noisy gradient descent over a schedule of shape stiffnesses mus.

    Every step all squares move at once by -lr * grad plus Gaussian noise of size temp * (1 - k/K)
    in position and angle. square=True ties a_i = b_i (squares may only shrink, not stretch).
    Sizes are clipped to [0.3, 0.5].
    """
    np.random.seed(seed)
    x, y, t, a, b = x.copy(), y.copy(), t.copy(), a.copy(), b.copy()
    n = x.shape[0]
    total = len(mus) * steps
    k = 0
    for mu in mus:
        lr = 0.2 / (1.0 + mu)  # stiff shape term needs smaller steps
        lr = max(lr, 0.002)
        for _ in range(steps):
            e, gx, gy, gt, ga, gb = energy_grad(x, y, t, a, b, L, mu, margin)
            T = temp * (1.0 - k / total)
            x -= 0.25 * gx + T * np.random.randn(n)
            y -= 0.25 * gy + T * np.random.randn(n)
            t -= 0.25 * gt + T * np.random.randn(n)
            if square:
                gs = 0.5 * (ga + gb)
                a -= lr * gs
                b[:] = a
            else:
                a -= lr * ga
                b -= lr * gb
            for i in range(n):
                a[i] = min(0.5, max(0.3, a[i]))
                b[i] = min(0.5, max(0.3, b[i]))
            k += 1
    return x, y, t, a, b


def soft_trial(p, L0, shrink, mus, steps, temp, square, seed, margin=MARGIN):
    """One soft-melt trial from packing p (rows x, y, angle) at side L0.

    Scales p into side L1 = L0 * (1 - shrink), melts and hardens it, then relaxes rigid unit squares
    at L1. Returns (x, y, t, L) of the smallest feasible packing found by squeezing from there, or
    None if no feasible packing exists at L1. Also returns the size deficit map (1 - side) after the melt.
    """
    L1 = L0 * (1 - shrink)
    k = L1 / L0
    x, y, t = p[:, 0] * k, p[:, 1] * k, p[:, 2].copy()
    n = len(x)
    a = np.full(n, 0.5)
    b = np.full(n, 0.5)
    mus = np.asarray(mus, float)
    # Melt at the softest stiffness with noise, then harden through the rest of the schedule.
    x, y, t, a, b = run(x, y, t, a, b, L1, mus[:1], 2 * steps, temp, square, seed, margin)
    deficit = 1.0 - 2 * np.minimum(a, b)
    x, y, t, a, b = run(x, y, t, a, b, L1, mus[1:], steps, 0.0, square, seed, margin)
    x, y, t, e = relax_lbfgs(x, y, t, L1, margin, 4000)
    if not feasible(x, y, t, L1, margin):
        return None, deficit
    r = squeeze(x, y, t, L1, rounds=30, step=1e-4, margin=margin, method="lbfgs", maxiter=2000)
    return r, deficit
