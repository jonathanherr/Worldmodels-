"""Drunk-walk grids: start square packings from a grid whose lines are random walks.

A plain packing of n unit squares puts them in the cells of an r x c grid. Here the inner grid
lines wander: horizontal line j is y_j(x) = j*L/r + W_j(x) and vertical line i is
x_i(y) = i*L/c + V_i(y), where W_j and V_i are random walks (steps of size `sigma` per unit
length, optionally smoothed). The outer lines stay on the walls. Each cell gets one square at
its centre, tilted to follow the local direction of the lines around it; `drop` cells (the
smallest ones) stay empty when r*c > n. The warped layout then goes to `pack_opt.compact_squeeze`,
which pulls it into a tight feasible packing.

The point is to start somewhere other than a record or a construction: every perturbation of a
record so far has snapped back into the record's basin, so these starts come from the grid side.
"""

import numpy as np

from .pack_opt import compact_squeeze


def walk(m, sigma, rng, smooth=0, pin_ends=False):
    """Random walk sampled at m points (unit spacing in grid cells), starting at 0."""
    w = np.concatenate([[0.0], np.cumsum(rng.normal(0.0, sigma, m - 1))])
    for _ in range(smooth):
        w = np.convolve(np.pad(w, 1, mode="edge"), [0.25, 0.5, 0.25], mode="valid")
    if pin_ends:
        w = w - np.linspace(w[0], w[-1], m)
    else:
        w = w - w.mean()
    return w


def drunk_lines(count, L, sigma, rng, samples, smooth=0, pin_ends=False):
    """count+1 lines across [0, L]: offsets (count+1, samples), outer two straight on the walls.

    Lines are clipped to the box and kept in order (they may touch but never cross).
    """
    base = np.linspace(0.0, L, count + 1)
    lines = np.repeat(base[:, None], samples, axis=1)
    for j in range(1, count):
        lines[j] += walk(samples, sigma, rng, smooth, pin_ends)
    lines = np.sort(np.clip(lines, 0.0, L), axis=0)
    lines[0], lines[-1] = 0.0, L
    return lines


def _interp(line, along, L):
    s = np.linspace(0.0, L, len(line))
    return np.interp(along, s, line), np.interp(along, s, np.gradient(line, s))


def drunk_grid(n, r, c, L, sigma, rng, samples=None, smooth=1, pin_ends=False, tilt=1.0, drop="small"):
    """Square centres and angles (x, y, radians) for n squares in a drunk-walk r x c grid of side L."""
    assert r * c >= n
    samples = samples or 4 * max(r, c) + 1
    H = drunk_lines(r, L, sigma, rng, samples, smooth, pin_ends)  # y offsets of horizontal lines vs x
    V = drunk_lines(c, L, sigma, rng, samples, smooth, pin_ends)  # x offsets of vertical lines vs y
    cx, cy, ang, area = [], [], [], []
    for j in range(r):
        for i in range(c):
            x0, y0 = (i + 0.5) * L / c, (j + 0.5) * L / r
            for _ in range(4):  # fixed point: centre between the two verticals and the two horizontals
                xa, sa = _interp(V[i], y0, L)
                xb, sb = _interp(V[i + 1], y0, L)
                ya, ta = _interp(H[j], x0, L)
                yb, tb = _interp(H[j + 1], x0, L)
                x0, y0 = 0.5 * (xa + xb), 0.5 * (ya + yb)
            # horizontals tilt by atan(dy/dx); verticals by -atan(dx/dy); a square follows their mean
            a = np.mean([np.arctan(ta), np.arctan(tb), -np.arctan(sa), -np.arctan(sb)])
            cx.append(x0), cy.append(y0), ang.append(tilt * a), area.append((xb - xa) * (yb - ya))
    cx, cy, ang, area = map(np.array, (cx, cy, ang, area))
    if drop == "small":
        keep = np.argsort(-area)[:n]
    else:
        keep = rng.choice(r * c, n, replace=False)
    return cx[keep], cy[keep], ang[keep]


def shapes(n, spread=1):
    """Grid shapes (r, c) with r*c >= n around ceil(sqrt n) and not more than k cells of spare."""
    k = int(np.ceil(np.sqrt(n)))
    out = []
    for r in range(k - spread, k + spread + 1):
        for c in range(r, k + spread + 1):
            if r >= 1 and r * c >= n and r * c - n < k:
                out.append((r, c))
    return out


def trial(n, r, c, sigma, seed, loose=1.08, smooth=1, pin_ends=False, tilt=1.0, drop="small", rounds=25):
    """One drunk-walk start relaxed by compact_squeeze. Returns (side, x, y, t) or None."""
    rng = np.random.default_rng(seed)
    L0 = max(r, c) * loose
    x, y, t = drunk_grid(n, r, c, L0, sigma, rng, smooth=smooth, pin_ends=pin_ends, tilt=tilt, drop=drop)
    res = compact_squeeze(x, y, t, L0, rounds=rounds)
    if res is None:
        return None
    x, y, t, L = res
    return float(L), x, y, t
