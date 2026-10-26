"""Helpers for program-search programs, in window-local coordinates (see wm/progsearch.py).

A square is (x, y, deg): centre and rotation in degrees. The box walls are x = 0 and y = 0;
the window is [0, size]^2 and squares outside it are frozen.

    block(x0, y0, rows, cols, deg)   rows x cols grid with its lower-left corner at (x0, y0),
                                      turned by deg about that corner
    rotate(squares, deg, about)      turn a list of squares rigidly
    translate(squares, dx, dy)
    worst_overlap(squares, frozen)   deepest overlap (<= 0 means none), walls included
    settle(squares, frozen)          push the squares apart (frozen ones stay put), keep the count
    waste_in(squares, frozen, size)  empty area left inside the window
"""

import math

import numpy as np
from scipy.optimize import minimize

from .pack_opt import MARGIN, energy_grad

BIG = 1000.0  # upper walls far away: only x = 0 and y = 0 constrain a window


def block(x0, y0, rows, cols, deg=0.0):
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    out = []
    for r in range(rows):
        for q in range(cols):
            u, v = q + 0.5, r + 0.5
            out.append((x0 + c * u - s * v, y0 + s * u + c * v, deg))
    return out


def rotate(squares, deg, about=None):
    if about is None:
        about = (sum(p[0] for p in squares) / len(squares), sum(p[1] for p in squares) / len(squares))
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    ox, oy = about
    return [(ox + c * (x - ox) - s * (y - oy), oy + s * (x - ox) + c * (y - oy), d + deg) for x, y, d in squares]


def translate(squares, dx, dy):
    return [(x + dx, y + dy, d) for x, y, d in squares]


def _arrays(squares, frozen):
    a = np.array(list(squares) + list(frozen), dtype=float).reshape(-1, 3)
    return a[:, 0] + BIG / 2, a[:, 1] + BIG / 2, np.radians(a[:, 2])


def worst_overlap(squares, frozen=()):
    """Deepest pair overlap or wall penetration (x < 0 or y < 0); <= 0 means the layout is clean."""
    x, y, t = _arrays(squares, frozen)
    *_, worst = energy_grad(x, y, t, 2 * BIG, MARGIN)
    a = np.array(list(squares), dtype=float).reshape(-1, 3)
    w = 0.5 * (np.abs(np.cos(np.radians(a[:, 2]))) + np.abs(np.sin(np.radians(a[:, 2]))))
    wall = float(np.max(np.concatenate([w - a[:, 0], w - a[:, 1]]))) if len(a) else -1.0
    return max(float(worst), wall)


def settle(squares, frozen=(), maxiter=2000):
    """Minimize overlap energy over the program's squares only; walls x = 0, y = 0 are enforced."""
    k = len(squares)
    sq = np.array(list(squares), dtype=float).reshape(-1, 3)
    fr = np.array(list(frozen), dtype=float).reshape(-1, 3)
    # Shift by BIG/2 so the far walls of the BIG box never bind, and add the two real walls as
    # penalties by hand (energy_grad only sees the BIG box).
    off = BIG / 2

    def f(z):
        x = np.concatenate([z[:k], fr[:, 0]]) + off
        y = np.concatenate([z[k:2 * k], fr[:, 1]]) + off
        t = np.concatenate([z[2 * k:], np.radians(fr[:, 2])])
        e, gx, gy, gt, _, _ = energy_grad(x, y, t, 2 * BIG, MARGIN)
        w = 0.5 * (np.abs(np.cos(z[2 * k:])) + np.abs(np.sin(z[2 * k:])))
        px, py = np.maximum(w + MARGIN - z[:k], 0), np.maximum(w + MARGIN - z[k:2 * k], 0)
        e += float((px ** 2).sum() + (py ** 2).sum())
        dw = 0.5 * (-np.sign(np.cos(z[2 * k:])) * np.sin(z[2 * k:]) + np.sign(np.sin(z[2 * k:])) * np.cos(z[2 * k:]))
        g = np.concatenate([gx[:k] - 2 * px, gy[:k] - 2 * py, gt[:k] + 2 * (px + py) * dw])
        return e, g

    z0 = np.concatenate([sq[:, 0], sq[:, 1], np.radians(sq[:, 2])])
    z = minimize(f, z0, jac=True, method="L-BFGS-B", options={"maxiter": maxiter, "ftol": 0.0, "gtol": 1e-14}).x
    return [(float(z[i]), float(z[k + i]), math.degrees(z[2 * k + i])) for i in range(k)]


def waste_in(squares, frozen, size, res=0.02):
    """Uncovered area of the window [0, size]^2 (raster estimate at resolution res)."""
    g = np.arange(res / 2, size, res)
    X, Y = np.meshgrid(g, g)
    cov = np.zeros_like(X, dtype=bool)
    for x, y, d in list(squares) + list(frozen):
        if abs(x - size / 2) > size / 2 + 1 or abs(y - size / 2) > size / 2 + 1:
            continue
        t = math.radians(d)
        u = math.cos(t) * (X - x) + math.sin(t) * (Y - y)
        v = -math.sin(t) * (X - x) + math.cos(t) * (Y - y)
        cov |= (np.abs(u) <= 0.5) & (np.abs(v) <= 0.5)
    return float((~cov).sum() * res * res)
