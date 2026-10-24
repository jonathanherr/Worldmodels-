"""Fast local optimizer for square packings: overlap energy, its gradient, and a shrink loop.

The energy of n unit squares (centers x, y, angles t in radians) in the box [0, L]^2 is

    E = sum over pairs of max(0, depth_ij)^2  +  sum over squares of wall penetration^2

where depth_ij is the separating-axis penetration depth (for two convex polygons the smallest
overlap over all edge normals is exactly how far one must move to separate them). With
d = c_j - c_i, delta = t_j - t_i and h(a) = |cos a| + |sin a|, the overlap along an edge normal
a of either square is 1/2 + h(delta)/2 - |d . a|, which gives a closed-form gradient.

`squeeze` finds a feasible packing at some side L, then repeatedly scales it down and
re-minimizes the energy, shrinking L while the packing stays feasible. The energy uses
squares inflated by `margin`, so a packing it calls feasible already has a real gap and
survives `wm.squares.polish` and Kaggle's interval-arithmetic check.
"""

import numpy as np
from numba import njit
from scipy.optimize import minimize

MARGIN = 2e-9


@njit(cache=True)
def _h(a):
    return abs(np.cos(a)) + abs(np.sin(a))


@njit(cache=True)
def _dh(a):
    return -np.sign(np.cos(a)) * np.sin(a) + np.sign(np.sin(a)) * np.cos(a)


@njit(cache=True)
def energy_grad(x, y, t, L, margin):
    """Energy, gradient (gx, gy, gt, gL) of a packing in [0, L]^2, and the worst pair depth."""
    n = x.shape[0]
    gx = np.zeros(n)
    gy = np.zeros(n)
    gt = np.zeros(n)
    e = 0.0
    gL = 0.0
    worst = -1.0
    c = np.cos(t)
    s = np.sin(t)
    reach = 2.0 + 4 * margin + 1e-9  # squared center distance beyond which two squares cannot touch
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[j] - x[i]
            dy = y[j] - y[i]
            if dx * dx + dy * dy >= reach:
                continue
            delta = t[j] - t[i]
            rad = 0.5 + 0.5 * _h(delta) + margin
            # The four edge normals: u_i, v_i, u_j, v_j, with u = (cos, sin), v = (-sin, cos).
            best = 1e9
            k_best = 0
            proj_best = 0.0
            for k in range(4):
                if k == 0:
                    ax, ay = c[i], s[i]
                elif k == 1:
                    ax, ay = -s[i], c[i]
                elif k == 2:
                    ax, ay = c[j], s[j]
                else:
                    ax, ay = -s[j], c[j]
                proj = dx * ax + dy * ay
                o = rad - abs(proj)
                if o < best:
                    best = o
                    k_best = k
                    proj_best = proj
            if best > worst:
                worst = best
            if best <= 0.0:
                continue
            e += best * best
            g = 2.0 * best
            sg = 1.0 if proj_best >= 0 else -1.0
            if k_best == 0:
                ax, ay, bx, by, owner = c[i], s[i], -s[i], c[i], i
            elif k_best == 1:
                ax, ay, bx, by, owner = -s[i], c[i], -c[i], -s[i], i
            elif k_best == 2:
                ax, ay, bx, by, owner = c[j], s[j], -s[j], c[j], j
            else:
                ax, ay, bx, by, owner = -s[j], c[j], -c[j], -s[j], j
            # depth = rad(delta) - sg * (d . a(t_owner)); (bx, by) = da/dt_owner.
            gx[j] += g * (-sg * ax)
            gy[j] += g * (-sg * ay)
            gx[i] -= g * (-sg * ax)
            gy[i] -= g * (-sg * ay)
            gt[owner] += g * (-sg * (dx * bx + dy * by))
            dr = 0.5 * _dh(delta)
            gt[j] += g * dr
            gt[i] -= g * dr
    for i in range(n):
        w = 0.5 * _h(t[i])
        dw = 0.5 * _dh(t[i])
        for coord in range(2):
            p = x[i] if coord == 0 else y[i]
            lo = w - p
            hi = p + w - L
            gp = 0.0
            if lo > 0:
                e += lo * lo
                gp -= 2 * lo
                gt[i] += 2 * lo * dw
            if hi > 0:
                e += hi * hi
                gp += 2 * hi
                gL -= 2 * hi
                gt[i] += 2 * hi * dw
            if coord == 0:
                gx[i] += gp
            else:
                gy[i] += gp
    return e, gx, gy, gt, gL, worst


@njit(cache=True)
def relax_fire(x, y, t, L, margin, maxiter):
    """Minimize the overlap energy at fixed L with FIRE (fast inertial relaxation), all in numba.

    The energy is exactly zero on the feasible set, so a feasible packing is reached in finitely
    many steps and the loop stops there. Returns (x, y, t, energy).
    """
    x, y, t = x.copy(), y.copy(), t.copy()
    n = x.shape[0]
    vx, vy, vt = np.zeros(n), np.zeros(n), np.zeros(n)
    dt, dt_max, alpha, n_pos = 0.02, 0.2, 0.1, 0
    e, gx, gy, gt, _, _ = energy_grad(x, y, t, L, margin)
    best_e, since_best = e, 0
    for _ in range(maxiter):
        if e == 0.0:
            break
        p = -(vx * gx).sum() - (vy * gy).sum() - (vt * gt).sum()
        vnorm = np.sqrt((vx * vx).sum() + (vy * vy).sum() + (vt * vt).sum())
        fnorm = np.sqrt((gx * gx).sum() + (gy * gy).sum() + (gt * gt).sum()) + 1e-300
        if p > 0:
            vx = (1 - alpha) * vx - alpha * vnorm * gx / fnorm
            vy = (1 - alpha) * vy - alpha * vnorm * gy / fnorm
            vt = (1 - alpha) * vt - alpha * vnorm * gt / fnorm
            n_pos += 1
            if n_pos > 5:
                dt = min(dt * 1.1, dt_max)
                alpha *= 0.99
        else:
            vx[:] = 0.0
            vy[:] = 0.0
            vt[:] = 0.0
            dt *= 0.5
            alpha = 0.1
            n_pos = 0
        vx -= dt * gx
        vy -= dt * gy
        vt -= dt * gt
        x += dt * vx
        y += dt * vy
        t += dt * vt
        e, gx, gy, gt, _, _ = energy_grad(x, y, t, L, margin)
        if e < best_e * (1 - 1e-4):
            best_e, since_best = e, 0
        else:
            since_best += 1
            if since_best > 300:  # stuck in a positive-energy minimum: infeasible at this L
                break
    return x, y, t, e


def relax_lbfgs(x, y, t, L, margin=MARGIN, maxiter=3000):
    """Minimize the overlap energy at fixed side L. Returns (x, y, t, energy)."""
    n = len(x)

    def f(z):
        e, gx, gy, gt, _, _ = energy_grad(z[:n], z[n:2 * n], z[2 * n:], L, margin)
        return e, np.concatenate([gx, gy, gt])

    z0 = np.concatenate([x, y, t]).astype(np.float64)
    res = minimize(f, z0, jac=True, method="L-BFGS-B",
                   options={"maxiter": maxiter, "ftol": 0.0, "gtol": 1e-16, "maxcor": 20})
    z = res.x
    return z[:n], z[n:2 * n], z[2 * n:], float(res.fun)


def compact(x, y, t, L, mus=(10.0, 100.0, 1e3, 1e4, 1e5), maxiter=400, margin=MARGIN):
    """Penalty method with the box side as a variable: minimize L + mu * E(x, y, t, L) for growing mu.

    Pulls a loose layout into a tight, nearly feasible one in a few hundred gradient steps; the
    result still overlaps by about 1/mu and goes to `squeeze` to become exactly feasible.
    """
    n = len(x)
    z = np.concatenate([x, y, t, [L]]).astype(np.float64)
    for mu in mus:
        def f(z):
            e, gx, gy, gt, gL, _ = energy_grad(z[:n], z[n:2 * n], z[2 * n:3 * n], z[-1], margin)
            return z[-1] + mu * e, np.concatenate([mu * gx, mu * gy, mu * gt, [1.0 + mu * gL]])

        z = minimize(f, z, jac=True, method="L-BFGS-B", options={"maxiter": maxiter, "maxcor": 20}).x
    return z[:n], z[n:2 * n], z[2 * n:3 * n], float(z[-1])


def feasible(x, y, t, L, margin=MARGIN) -> bool:
    """True if no pair overlaps (with the inflated squares) and every square is inside [0, L]^2."""
    *_, worst = energy_grad(x, y, t, L, margin)
    w = 0.5 * (np.abs(np.cos(t)) + np.abs(np.sin(t)))
    inside = np.all(x - w >= -1e-12) and np.all(x + w <= L + 1e-12) and np.all(y - w >= -1e-12) and np.all(y + w <= L + 1e-12)
    return worst <= 0.0 and inside


def _scale(x, y, L_from, L_to):
    k = L_to / L_from
    return x * k, y * k


def compact_squeeze(x, y, t, L, rounds=25, maxiter=400, margin=MARGIN):
    """`compact` a loose layout, then `squeeze` it to an exactly feasible packing."""
    x, y, t, L = compact(x, y, t, L, maxiter=maxiter, margin=margin)
    return squeeze(x, y, t, L * 1.0005, rounds=rounds, step=0.001, margin=margin, method="lbfgs",
                   maxiter=1500)


def squeeze(x, y, t, L, rounds=40, step=0.03, min_step=1e-7, maxiter=3000, margin=MARGIN, method="fire"):
    """Shrink the box around a packing while it stays feasible.

    Starts at side L (grows it until the packing relaxes to feasible), then tries L - step;
    on success it keeps the smaller box and grows the step, on failure it halves the step.
    Returns (x, y, t, L) of the smallest feasible packing found. method is "fire" (numba, fast)
    or "lbfgs" (scipy, converges tighter on large n but has per-iteration Python overhead).
    """
    relax = relax_fire if method == "fire" else relax_lbfgs
    x, y, t = (np.array(a, dtype=np.float64) for a in (x, y, t))
    for _ in range(60):
        x, y, t, _ = relax(x, y, t, L, margin, maxiter)
        if feasible(x, y, t, L, margin):
            break
        x, y = _scale(x, y, L, L * 1.03)
        L *= 1.03
    else:
        return None
    best = (x, y, t, L)
    step *= L
    for _ in range(rounds):
        if step < min_step:
            break
        bx, by, bt, bL = best
        L_try = bL - step
        sx, sy = _scale(bx, by, bL, L_try)
        nx, ny, nt, _ = relax(sx, sy, bt, L_try, margin, maxiter)
        if feasible(nx, ny, nt, L_try, margin):
            best = (nx, ny, nt, L_try)
            step *= 1.5
        else:
            step *= 0.4
    return best
