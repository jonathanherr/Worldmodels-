"""Dimensional relaxation: let squares pass over each other in a third dimension, then flatten.

Each unit square gets a height z and becomes a slab of thickness `h` (a square prism). Two
slabs overlap only if their footprints overlap *and* |z_i - z_j| < h, so the 3D penetration
depth of a pair is

    p_ij = min(depth2d_ij, h - |z_i - z_j|)

(the z axis is just one more separating axis for prisms). The energy is sum max(0, p_ij)^2
plus wall penetration in x, y and in z for the height box [0, H]. With H >= h a jammed square
can climb over its neighbours instead of having to push them; annealing H to 0 squeezes all
heights back into the plane, where the energy is exactly the 2D overlap energy of
`wm.pack_opt`, so a zero-energy end state is a valid 2D packing.

`lift_compress` uses this to try a smaller box than a known packing: scale the packing down
to the target side, lift the squares to random heights, relax, anneal H -> 0, finish with the
2D optimizer and `squeeze`. Passing H0=0 runs the same pipeline flat, as a control.
"""

import numpy as np
from numba import njit

from scipy.optimize import minimize

from .pack_opt import MARGIN, _dh, _h, compact, feasible, relax_fire, squeeze


@njit(cache=True)
def energy_grad_lift(x, y, t, z, L, H, h, kappa, lens, margin):
    """Energy, gradient (gx, gy, gt, gz, gL) of lifted squares, and the worst lifted pair depth.

    lens=False: slabs of thickness h in the height box [0, H]. lens=True: pair depth
    depth2d * (1 - |dz| / h), heights held by the spring kappa * z^2 instead of a box.
    """
    n = x.shape[0]
    gx = np.zeros(n)
    gy = np.zeros(n)
    gt = np.zeros(n)
    gz = np.zeros(n)
    e = 0.0
    gL = 0.0
    worst = -1.0
    c = np.cos(t)
    s = np.sin(t)
    reach = 2.0 + 4 * margin + 1e-9
    for i in range(n):
        for j in range(i + 1, n):
            dz = z[j] - z[i]
            zov = h - abs(dz)
            if zov <= 0.0:
                continue
            dx = x[j] - x[i]
            dy = y[j] - y[i]
            if dx * dx + dy * dy >= reach:
                continue
            delta = t[j] - t[i]
            rad = 0.5 + 0.5 * _h(delta) + margin
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
            if lens:  # overlap fades linearly with height difference: p = depth2d * zov / h
                if best <= 0.0:
                    continue
                f = zov / h
                p = best * f
                if p > worst:
                    worst = p
                e += p * p
                g = 2.0 * p * f
                sz = 1.0 if dz >= 0 else -1.0
                gz[j] -= 2.0 * p * best / h * sz
                gz[i] += 2.0 * p * best / h * sz
            elif zov < best:  # slab: separating along z is cheapest, push the pair apart vertically
                if zov > worst:
                    worst = zov
                e += zov * zov
                g = 2.0 * zov
                sz = 1.0 if dz >= 0 else -1.0
                gz[j] -= g * sz
                gz[i] += g * sz
                continue
            else:
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
        if lens:
            e += kappa * z[i] * z[i]
            gz[i] += 2 * kappa * z[i]
        elif z[i] < 0.0:
            e += z[i] * z[i]
            gz[i] += 2 * z[i]
        elif z[i] > H:
            e += (z[i] - H) ** 2
            gz[i] += 2 * (z[i] - H)
    return e, gx, gy, gt, gz, gL, worst


@njit(cache=True)
def relax_fire_lift(x, y, t, z, L, H, h, kappa, lens, margin, maxiter, patience=300):
    """FIRE minimization of the lifted energy at fixed L and H. Returns (x, y, t, z, energy)."""
    x, y, t, z = x.copy(), y.copy(), t.copy(), z.copy()
    n = x.shape[0]
    vx, vy, vt, vz = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    dt, dt_max, alpha, n_pos = 0.02, 0.2, 0.1, 0
    e, gx, gy, gt, gz, _, _ = energy_grad_lift(x, y, t, z, L, H, h, kappa, lens, margin)
    best_e, since_best = e, 0
    for _ in range(maxiter):
        if e == 0.0:
            break
        p = -(vx * gx).sum() - (vy * gy).sum() - (vt * gt).sum() - (vz * gz).sum()
        vnorm = np.sqrt((vx * vx).sum() + (vy * vy).sum() + (vt * vt).sum() + (vz * vz).sum())
        fnorm = np.sqrt((gx * gx).sum() + (gy * gy).sum() + (gt * gt).sum() + (gz * gz).sum()) + 1e-300
        if p > 0:
            vx = (1 - alpha) * vx - alpha * vnorm * gx / fnorm
            vy = (1 - alpha) * vy - alpha * vnorm * gy / fnorm
            vt = (1 - alpha) * vt - alpha * vnorm * gt / fnorm
            vz = (1 - alpha) * vz - alpha * vnorm * gz / fnorm
            n_pos += 1
            if n_pos > 5:
                dt = min(dt * 1.1, dt_max)
                alpha *= 0.99
        else:
            vx[:] = 0.0
            vy[:] = 0.0
            vt[:] = 0.0
            vz[:] = 0.0
            dt *= 0.5
            alpha = 0.1
            n_pos = 0
        vx -= dt * gx
        vy -= dt * gy
        vt -= dt * gt
        vz -= dt * gz
        x += dt * vx
        y += dt * vy
        t += dt * vt
        z += dt * vz
        e, gx, gy, gt, gz, _, _ = energy_grad_lift(x, y, t, z, L, H, h, kappa, lens, margin)
        if e < best_e * (1 - 1e-4):
            best_e, since_best = e, 0
        else:
            since_best += 1
            if since_best > patience:
                break
    return x, y, t, z, e


def flatten(x, y, t, z, L, H0=1.5, h=1.0, lens=True, kappa0=1e-6, kappa_mult=4.0, stages=12,
            maxiter=3000, margin=MARGIN):
    """Push the heights back to the plane in stages, relaxing at each, then relax in 2D.

    lens: the spring constant grows from kappa0 by kappa_mult per stage. slab: the height box
    shrinks from H0 to 0, with heights rescaled into it before each stage. Returns
    (x, y, t, energy) of the final 2D relaxation; energy 0 means feasible at side L.
    """
    for k in range(1, stages + 1):
        if lens:
            kappa, H = kappa0 * kappa_mult ** k, 0.0
        else:
            kappa, H = 0.0, H0 * (1 - k / stages) ** 2
            zmax = z.max()
            if zmax > H and zmax > 0:
                z = z * (H / zmax)
        x, y, t, z, _ = relax_fire_lift(x, y, t, z, L, H, h, kappa, lens, margin, maxiter)
    return relax_fire(x, y, t, L, margin, maxiter)


def lift_compress(x, y, t, L_from, L_to, H0=1.0, h=1.0, lens=True, kappa0=1e-6, kappa_mult=4.0,
                  stages=12, frac=1.0, jitter=0.0, maxiter=3000, rng=None, margin=MARGIN):
    """Try to fit a packing at side L_from into side L_to by lifting, relaxing and flattening.

    frac of the squares (chosen at random) start at random heights in [-H0, H0] (lens) or
    [0, H0] (slab); the rest start in the plane. H0=0 is the flat control (scale + 2D
    relaxation only). Returns (x, y, t) if the result is feasible at L_to, else None.
    """
    rng = rng or np.random.default_rng()
    n = len(x)
    k = L_to / L_from
    x, y, t = np.asarray(x, float) * k, np.asarray(y, float) * k, np.asarray(t, float).copy()
    if jitter:
        x += rng.normal(0, jitter, n)
        y += rng.normal(0, jitter, n)
        t += rng.normal(0, jitter, n)
    if H0 > 0:
        z = np.zeros(n)
        lifted = rng.random(n) < frac
        z[lifted] = rng.uniform(-H0 if lens else 0.0, H0, lifted.sum())
        x, y, t, e = flatten(x, y, t, z, L_to, H0, h, lens, kappa0, kappa_mult, stages, maxiter, margin)
    else:
        x, y, t, e = relax_fire(x, y, t, L_to, margin, maxiter)
    if e == 0.0 or feasible(x, y, t, L_to, margin):
        return x, y, t
    return None

def lift_search(x, y, t, L, tries=20, shrink=0.004, min_shrink=2e-5, rng=None, log=None, **kw):
    """Repeatedly ask `lift_compress` for a box `shrink` (relative) smaller than the current best.

    On success the new packing is squeezed further with the 2D optimizer and becomes the base;
    on failure the shrink is halved every few tries. Returns (x, y, t, L) of the best packing.
    """
    rng = rng or np.random.default_rng()
    best = (np.asarray(x, float), np.asarray(y, float), np.asarray(t, float), float(L))
    fails = 0
    for i in range(tries):
        if shrink < min_shrink:
            break
        bx, by, bt, bL = best
        L_to = bL * (1 - shrink)
        r = lift_compress(bx, by, bt, bL, L_to, rng=rng, **kw)
        if r is not None:
            sq = squeeze(*r, L_to, rounds=30, step=shrink / 4)
            best = sq if sq is not None and sq[3] <= L_to else (*r, L_to)
            if log:
                log(f"  try {i}: fits {best[3]:.6f} (asked {L_to:.6f})")
            fails = 0
        else:
            fails += 1
            if fails >= 3:
                shrink /= 2
                fails = 0
    return best


def lift_compact(x, y, t, L, z=None, h=1.0, kappas=(0.1, 1.0, 10.0),
                 mus=(10.0, 100.0, 1e3, 1e4), maxiter=400, margin=MARGIN):
    """`pack_opt.compact` with lifted squares: minimize L + mu * E_lens for growing mu while the
    height spring kappa stiffens, so squares can slide past each other while the box shrinks.

    Ends with the flat `compact`, so the result is in the plane. Returns (x, y, t, L).
    """
    n = len(x)
    if z is None:
        z = np.random.default_rng().uniform(-0.5, 0.5, n)
    v = np.concatenate([x, y, t, z, [L]]).astype(np.float64)
    for kappa in kappas:
        for mu in mus:
            def f(v):
                e, gx, gy, gt, gz, gL, _ = energy_grad_lift(v[:n], v[n:2 * n], v[2 * n:3 * n], v[3 * n:4 * n],
                                                            v[-1], 0.0, h, kappa, True, margin)
                return v[-1] + mu * e, np.concatenate([mu * gx, mu * gy, mu * gt, mu * gz, [1.0 + mu * gL]])

            v = minimize(f, v, jac=True, method="L-BFGS-B", options={"maxiter": maxiter, "maxcor": 20}).x
    return compact(v[:n], v[n:2 * n], v[2 * n:3 * n], float(v[-1]), maxiter=maxiter, margin=margin)
