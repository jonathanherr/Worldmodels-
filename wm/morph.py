"""Neighbouring-n and fractional-n packing: build a packing for n by morphing one for n +- 1.

Two moves, each with a continuous ("fractional") and an abrupt version:

- remove (n+1 -> n): drop one square of the n+1 packing and recompress. The fractional form
  shrinks that square's size s_r from 1 to 0 while the box side L is a free variable, so the
  rest of the packing flows into the space as it opens instead of being left with a hole.
- insert (n -> n+1): put a small square into the roomiest hole of the n packing and grow its
  size from s0 to 1, again with L free, so the neighbours make room gradually.

Fractional n is the sum of sizes: n packings with one partial square interpolate between
the n and n+1 problems. Everything runs on a size-aware copy of pack_opt's separating-axis
overlap energy; once all sizes are 1, pack_opt.squeeze finishes the packing to an exactly
feasible one that best_store can certify.
"""

import numpy as np
from numba import njit
from scipy.optimize import minimize

from .pack_opt import MARGIN, compact, squeeze


@njit(cache=True)
def _h(a):
    return abs(np.cos(a)) + abs(np.sin(a))


@njit(cache=True)
def _dh(a):
    return -np.sign(np.cos(a)) * np.sin(a) + np.sign(np.sin(a)) * np.cos(a)


@njit(cache=True)
def energy_grad_sized(x, y, t, s, L, margin):
    """pack_opt.energy_grad for squares of side s[i] (s[i] = 0 means absent). Returns (e, gx, gy, gt, gL)."""
    n = x.shape[0]
    gx = np.zeros(n)
    gy = np.zeros(n)
    gt = np.zeros(n)
    e = 0.0
    gL = 0.0
    c = np.cos(t)
    sn = np.sin(t)
    for i in range(n):
        if s[i] <= 0.0:
            continue
        for j in range(i + 1, n):
            if s[j] <= 0.0:
                continue
            dx = x[j] - x[i]
            dy = y[j] - y[i]
            r = (s[i] + s[j]) / np.sqrt(2.0) + 2 * margin + 1e-9
            if dx * dx + dy * dy >= r * r:
                continue
            delta = t[j] - t[i]
            hd = _h(delta)
            best = 1e9
            k_best = 0
            proj_best = 0.0
            for k in range(4):
                if k == 0:
                    ax, ay, rad = c[i], sn[i], 0.5 * s[i] + 0.5 * s[j] * hd
                elif k == 1:
                    ax, ay, rad = -sn[i], c[i], 0.5 * s[i] + 0.5 * s[j] * hd
                elif k == 2:
                    ax, ay, rad = c[j], sn[j], 0.5 * s[j] + 0.5 * s[i] * hd
                else:
                    ax, ay, rad = -sn[j], c[j], 0.5 * s[j] + 0.5 * s[i] * hd
                proj = dx * ax + dy * ay
                o = rad + margin - abs(proj)
                if o < best:
                    best = o
                    k_best = k
                    proj_best = proj
            if best <= 0.0:
                continue
            e += best * best
            g = 2.0 * best
            sg = 1.0 if proj_best >= 0 else -1.0
            if k_best == 0:
                ax, ay, bx, by, owner, other = c[i], sn[i], -sn[i], c[i], i, j
            elif k_best == 1:
                ax, ay, bx, by, owner, other = -sn[i], c[i], -c[i], -sn[i], i, j
            elif k_best == 2:
                ax, ay, bx, by, owner, other = c[j], sn[j], -sn[j], c[j], j, i
            else:
                ax, ay, bx, by, owner, other = -sn[j], c[j], -c[j], -sn[j], j, i
            gx[j] += g * (-sg * ax)
            gy[j] += g * (-sg * ay)
            gx[i] -= g * (-sg * ax)
            gy[i] -= g * (-sg * ay)
            gt[owner] += g * (-sg * (dx * bx + dy * by))
            dr = 0.5 * s[other] * _dh(delta)
            gt[j] += g * dr
            gt[i] -= g * dr
    for i in range(n):
        if s[i] <= 0.0:
            continue
        w = 0.5 * s[i] * _h(t[i])
        dw = 0.5 * s[i] * _dh(t[i])
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
    return e, gx, gy, gt, gL


def compact_sized(x, y, t, s, L, mus=(1e3, 1e4, 1e5), maxiter=200, margin=MARGIN):
    """pack_opt.compact with fixed sizes s: minimize L + mu * E over (x, y, t, L) for growing mu."""
    n = len(x)
    s = np.asarray(s, dtype=np.float64)
    z = np.concatenate([x, y, t, [L]]).astype(np.float64)
    for mu in mus:
        def f(z):
            e, gx, gy, gt, gL = energy_grad_sized(z[:n], z[n:2 * n], z[2 * n:3 * n], s, z[-1], margin)
            return z[-1] + mu * e, np.concatenate([mu * gx, mu * gy, mu * gt, [1.0 + mu * gL]])

        z = minimize(f, z, jac=True, method="L-BFGS-B", options={"maxiter": maxiter, "maxcor": 20}).x
    return z[:n], z[n:2 * n], z[2 * n:3 * n], float(z[-1])


def _finish(x, y, t, L):
    """Exactly feasible packing (all sizes 1) from a nearly feasible one, or None."""
    return squeeze(x, y, t, L * 1.0005, rounds=30, step=0.001, method="lbfgs", maxiter=1500)


def remove_abrupt(x, y, t, L, r):
    """Drop square r and recompress the remaining packing (the non-fractional control)."""
    keep = np.arange(len(x)) != r
    x, y, t = x[keep], y[keep], t[keep]
    x, y, t, L2 = compact(x, y, t, L, mus=(1e3, 1e4, 1e5), maxiter=300)
    return _finish(x, y, t, L2)


def remove_fractional(x, y, t, L, r, steps=8, rng=None, jitter=0.0):
    """Shrink square r from size 1 to 0 in `steps` stages, compacting with L free at each, then drop it."""
    n = len(x)
    x, y, t = x.copy(), y.copy(), t.copy()
    s = np.ones(n)
    for k in range(1, steps + 1):
        s[r] = 1.0 - k / steps
        if jitter and rng is not None:
            x += rng.normal(0, jitter, n)
            y += rng.normal(0, jitter, n)
        x, y, t, L = compact_sized(x, y, t, s, L)
    keep = np.arange(n) != r
    return _finish(x[keep], y[keep], t[keep], L)


def clearance_map(x, y, t, L, res=0.05):
    """Free space at grid points of [0, L]^2: the largest axis-aligned half-size a square centred there
    could have without touching a wall (exact) or another square (Chebyshev distance in the
    neighbour's frame, which is exact for axis-aligned neighbours and a heuristic otherwise)."""
    g = np.arange(res / 2, L, res)
    px, py = np.meshgrid(g, g, indexing="ij")
    px, py = px.ravel(), py.ravel()
    clr = np.minimum.reduce([px, py, L - px, L - py])
    c, sn = np.cos(t), np.sin(t)
    for i in range(len(x)):
        dx, dy = px - x[i], py - y[i]
        u, v = dx * c[i] + dy * sn[i], -dx * sn[i] + dy * c[i]
        d = np.maximum(np.abs(u), np.abs(v)) - 0.5
        d = np.where(d > 0, d / _h_np(t[i]), d)
        clr = np.minimum(clr, d)
    return px, py, clr


def _h_np(a):
    return abs(np.cos(a)) + abs(np.sin(a))


def holes(x, y, t, L, k=4, res=0.05):
    """Up to k well-separated roomiest points, as (cx, cy, half_size)."""
    px, py, clr = clearance_map(x, y, t, L, res)
    order = np.argsort(-clr)
    out = []
    for idx in order:
        if clr[idx] <= 0:
            break
        if all((px[idx] - a) ** 2 + (py[idx] - b) ** 2 > 1.0 for a, b, _ in out):
            out.append((px[idx], py[idx], clr[idx]))
            if len(out) == k:
                break
    return out


def insert_fractional(x, y, t, L, hole, steps=8, angle=0.0):
    """Add a square at hole = (cx, cy, half) with starting size 2*half (capped at 1), grow it to 1."""
    cx, cy, half = hole
    n = len(x)
    x, y, t = np.append(x, cx), np.append(y, cy), np.append(t, angle)
    s = np.ones(n + 1)
    s0 = min(1.0, max(0.02, 2 * half * 0.95))
    for k in range(steps + 1):
        s[n] = s0 + (1.0 - s0) * k / steps
        x, y, t, L = compact_sized(x, y, t, s, L)
    return _finish(x, y, t, L)


def insert_abrupt(x, y, t, L, hole, angle=0.0):
    """Add a full-size square at the hole and let compact push everything apart (the control)."""
    cx, cy, _ = hole
    x, y, t = np.append(x, cx), np.append(y, cy), np.append(t, angle)
    x, y, t, L = compact(x, y, t, L, mus=(1e3, 1e4, 1e5), maxiter=300)
    return _finish(x, y, t, L)


def removal_candidates(x, y, t, L, k=8, rng=None):
    """Squares worth trying to remove: the ones touching the walls that set the box side, plus random picks.
    A square on the top or right wall props that wall up, so removing it can let the box shrink."""
    w = 0.5 * (np.abs(np.cos(t)) + np.abs(np.sin(t)))
    lo = np.minimum(x - w, y - w)
    hi = np.maximum(x + w, y + w)
    slack = np.minimum(lo, L - hi)
    order = list(np.argsort(slack))
    picks = order[: k // 2]
    rng = rng or np.random.default_rng(0)
    rest = [i for i in range(len(x)) if i not in picks]
    picks += list(rng.choice(rest, size=min(k - len(picks), len(rest)), replace=False))
    return [int(i) for i in picks]


# ---------------------------------------------------------------------------------------------
# Sub-divided n: a packing for n = a + b + 2 * fill(s_a, s_b) built from the stored packings of
# a and b. Block a sits in one corner, block b in the opposite one, and the two s_a x s_b
# rectangles left over are filled with the best of a unit grid or (smaller stored square
# packing + grid strip). This is how improvements to one n propagate to larger n: each time a
# solver lowers s(a), every composition that uses a gets cheaper too.

BLOCK_GAP = 1e-8  # between blocks, so squares of neighbouring blocks never touch exactly


def _grid(w, h, x0, y0):
    """Unit grid in the w x h rectangle, spaced a hair over 1 so neighbours never touch exactly."""
    kx, ky = int(np.floor(w + 1e-9)), int(np.floor(h + 1e-9))
    ex = min(1e-8, (w - kx) / (kx + 1)) if kx else 0.0
    ey = min(1e-8, (h - ky) / (ky + 1)) if ky else 0.0
    i, j = np.meshgrid(np.arange(kx), np.arange(ky), indexing="ij")
    return (x0 + (i.ravel() + 1) * ex + i.ravel() + 0.5, y0 + (j.ravel() + 1) * ey + j.ravel() + 0.5,
            np.zeros(kx * ky))


def rect_fill(w, h, sides, packings, x0=0.0, y0=0.0):
    """Squares in the w x h rectangle at (x0, y0): a unit grid, or the largest stored square packing
    that fits in the short side plus a grid in the leftover strip, whichever holds more.
    sides[m] is the stored side of m (nondecreasing); packings(m) returns (x, y, t, L) in [0, L]^2."""
    best = _grid(w, h, x0, y0)
    a, c = min(w, h), max(w, h)
    m = int(np.searchsorted(sides[1:], a - BLOCK_GAP, side="right"))
    strip = int(np.floor(c - sides[m] - BLOCK_GAP + 1e-9)) * int(np.floor(a + 1e-9)) if m else 0
    if m and m + strip > len(best[0]):
        px, py, pt, pL = packings(m)
        if w >= h:
            sx, sy, st = _grid(w - pL - BLOCK_GAP, h, x0 + pL + BLOCK_GAP, y0)
            best = (np.concatenate([px + x0, sx]), np.concatenate([py + y0, sy]), np.concatenate([pt, st]))
        else:
            sx, sy, st = _grid(w, h - pL - BLOCK_GAP, x0, y0 + pL + BLOCK_GAP)
            best = (np.concatenate([px + x0, sx]), np.concatenate([py + y0, sy]), np.concatenate([pt, st]))
    return best


def rect_count(w, h, sides):
    a, c = min(w, h), max(w, h)
    g = int(np.floor(w + 1e-9)) * int(np.floor(h + 1e-9))
    m = int(np.searchsorted(sides[1:], a - BLOCK_GAP, side="right"))
    strip = int(np.floor(c - sides[m] - BLOCK_GAP + 1e-9)) * int(np.floor(a + 1e-9)) if m else 0
    return max(g, m + strip)


def compose_2x2(a, b, sides, packings):
    """Packing from blocks a and b in opposite corners plus two filled s_a x s_b rectangles.
    Returns (x, y, t, S) with all n = a + b + 2 * rect_count(s_a, s_b) squares."""
    ax, ay, at, sa = packings(a)
    bx, by, bt, sb = packings(b)
    o = sa + BLOCK_GAP
    S = o + sb
    r1 = rect_fill(sb, sa, sides, packings, o, 0.0)
    r2 = rect_fill(sa, sb, sides, packings, 0.0, o)
    x = np.concatenate([ax, bx + o, r1[0], r2[0]])
    y = np.concatenate([ay, by + o, r1[1], r2[1]])
    t = np.concatenate([at, bt, r1[2], r2[2]])
    return x, y, t, S


def composition_wins(sides, max_n=1000, margin=1e-6):
    """{n: (S, a, b)} for every n where some 2x2 composition beats sides[n] (sides nondecreasing)."""
    wins = {}
    for a in range(1, max_n + 1):
        for b in range(a, max_n + 1):
            S = sides[a] + sides[b] + BLOCK_GAP
            if S > sides[max_n] + 1:
                break
            n = a + b + 2 * rect_count(sides[a], sides[b], sides)
            if n <= max_n and S < sides[n] - margin and (n not in wins or S < wins[n][0]):
                wins[n] = (S, a, b)
    return wins


def breathe(x, y, t, L, rng, frac=0.3, s_min=0.7, steps=6, jitter=0.02):
    """Fractional-size perturbation of a packing: shrink a random patch of squares to s_min, compact with
    L free so the rest flows into the room, then grow them back to 1 step by step. Unlike scaling the
    whole box this changes the packing's shape, so it can leave a jammed basin. Returns squeeze's result."""
    n = len(x)
    c = rng.integers(n)
    d2 = (x - x[c]) ** 2 + (y - y[c]) ** 2
    patch = np.argsort(d2)[: max(1, int(frac * n))]
    s = np.ones(n)
    x = x + rng.normal(0, jitter, n)
    y = y + rng.normal(0, jitter, n)
    t = t + rng.normal(0, jitter, n)
    for k in range(steps + 1):
        s[patch] = s_min + (1 - s_min) * k / steps
        x, y, t, L = compact_sized(x, y, t, s, L)
    return _finish(x, y, t, L)
