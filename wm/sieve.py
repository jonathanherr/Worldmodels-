"""Sieve-and-settle: granular deposition of unit squares through a shaken mesh, then box closing.

The squares are rigid bodies with soft contacts (the overlap energy of `wm.pack_opt`, scaled by
a stiffness k) moving under gravity, damping and random kicks (a temperature). Three scenarios
share the same integrator, so their packings can be compared at equal simulation budget:

- `sieve`: squares wait in a hopper above a horizontal mesh plate whose holes are wider than one
  square but narrower than its diagonal. The plate is shaken sideways; a square only drops
  through when it lines up with a hole at a passable angle, so the pile below is fed one hole
  column at a time, in a partly ordered stream.
- `cells`: the box starts divided by thin walls into a coarse lattice of cells (the "larger
  squares"). Squares are dropped into cells and settle inside them; then the walls thin out
  and vanish while the box is shaken, so the small groups merge (a multiscale settle).
- `shake`: the control. Same hopper and the same shaking/tapping schedule, no mesh or walls.

After settling, the pile is closed into a square box with `pack_opt.compact_squeeze`, which
returns an exactly feasible packing with a small gap (passes Kaggle's interval check).
"""

import numpy as np
from numba import njit

from .pack_opt import _dh, _h, compact_squeeze, energy_grad, feasible

BIG = 1e6


@njit(cache=True)
def bars_energy_grad(x, y, t, bars, gx, gy, gt):
    """Overlap energy of squares with fixed axis-aligned bars (cx, cy, hx, hy); adds to gradients.

    Separating-axis depth over the four normals x, y, u, v (u, v = the square's own axes).
    """
    e = 0.0
    n = x.shape[0]
    for b in range(bars.shape[0]):
        cx, cy, hx, hy = bars[b, 0], bars[b, 1], bars[b, 2], bars[b, 3]
        if hx <= 0.0 or hy <= 0.0:
            continue
        for i in range(n):
            dx = x[i] - cx
            dy = y[i] - cy
            if abs(dx) > hx + 0.75 or abs(dy) > hy + 0.75:
                continue
            c = np.cos(t[i])
            s = np.sin(t[i])
            ac, as_ = abs(c), abs(s)
            # axis x
            best = 0.5 * (ac + as_) + hx - abs(dx)
            k = 0
            # axis y
            o = 0.5 * (ac + as_) + hy - abs(dy)
            if o < best:
                best, k = o, 1
            pu = dx * c + dy * s
            pv = -dx * s + dy * c
            o = 0.5 + hx * ac + hy * as_ - abs(pu)
            if o < best:
                best, k = o, 2
            o = 0.5 + hx * as_ + hy * ac - abs(pv)
            if o < best:
                best, k = o, 3
            if best <= 0.0:
                continue
            e += best * best
            g = 2.0 * best
            if k == 0:
                sg = 1.0 if dx >= 0 else -1.0
                gx[i] += g * (-sg)
                gt[i] += g * 0.5 * _dh(t[i])
            elif k == 1:
                sg = 1.0 if dy >= 0 else -1.0
                gy[i] += g * (-sg)
                gt[i] += g * 0.5 * _dh(t[i])
            elif k == 2:
                sg = 1.0 if pu >= 0 else -1.0
                gx[i] += g * (-sg * c)
                gy[i] += g * (-sg * s)
                dr = -hx * np.sign(c) * s + hy * np.sign(s) * c
                gt[i] += g * (dr - sg * pv)
            else:
                sg = 1.0 if pv >= 0 else -1.0
                gx[i] += g * (sg * s)
                gy[i] += g * (-sg * c)
                dr = hx * np.sign(s) * c - hy * np.sign(c) * s
                gt[i] += g * (dr + sg * pu)
    return e


@njit(cache=True)
def simulate(x, y, t, vx, vy, vt, W, H, bars, steps, k, gdir_x, gdir_y, g, temp, dt, damp, seed):
    """Damped Langevin rigid-body dynamics in a container [0, W] x [0, H]. Updates in place.

    Forces: -k * grad(overlap energy) (pairs, walls, bars) + gravity g * (gdir_x, gdir_y)
    + Gaussian kicks of size temp * sqrt(dt). Rotation has the unit square's inertia (1/6).
    """
    np.random.seed(seed)
    n = x.shape[0]
    kick = temp * np.sqrt(dt)
    vmax = 0.2 / dt
    for _ in range(steps):
        _, gx, gy, gt, _, _ = energy_grad(x, y, t, BIG, 0.0)
        bars_energy_grad(x, y, t, bars, gx, gy, gt)
        for i in range(n):
            hi = x[i] + 0.5 * _h(t[i]) - W
            if hi > 0:
                gx[i] += 2 * hi
                gt[i] += 2 * hi * 0.5 * _dh(t[i])
            hi = y[i] + 0.5 * _h(t[i]) - H
            if hi > 0:
                gy[i] += 2 * hi
                gt[i] += 2 * hi * 0.5 * _dh(t[i])
            vx[i] = vx[i] * (1 - damp) + dt * (-k * gx[i] + g * gdir_x) + kick * np.random.randn()
            vy[i] = vy[i] * (1 - damp) + dt * (-k * gy[i] + g * gdir_y) + kick * np.random.randn()
            vt[i] = vt[i] * (1 - damp) + dt * 6.0 * (-k * gt[i]) + 2.0 * kick * np.random.randn()
            # Velocity cap: a safety net against a stiff contact kicking a square through a wall.
            vx[i] = min(max(vx[i], -vmax), vmax)
            vy[i] = min(max(vy[i], -vmax), vmax)
            vt[i] = min(max(vt[i], -4 * vmax), 4 * vmax)
            x[i] += dt * vx[i]
            y[i] += dt * vy[i]
            t[i] += dt * vt[i]


def _hopper(n, W, y0, rng, spacing=1.5):
    """Loose non-overlapping rows above height y0 (spacing > sqrt 2), random angles."""
    per = max(1, int((W - 0.2) // spacing))
    i = np.arange(n)
    x = 0.1 + spacing * (i % per + 0.5) + rng.uniform(-0.03, 0.03, n)
    y = y0 + spacing * (i // per + 0.5)
    t = rng.uniform(-np.pi / 4, np.pi / 4, n)
    return x, y, t


def _plate(W, yp, hole, pitch, offset, thick=0.15):
    """Bars of a mesh plate at height yp: holes of width `hole` every `pitch`, shifted by offset."""
    bars = []
    m = max(1, int(round(W / pitch)))
    pitch = W / m
    edges = []
    for j in range(-1, m + 1):
        c = (j + 0.5) * pitch + offset
        edges.append((c - hole / 2, c + hole / 2))
    for (a0, a1), (b0, b1) in zip(edges[:-1], edges[1:]):
        lo, hi = max(a1, -1.0), min(b0, W + 1.0)
        if hi > lo:
            bars.append(((lo + hi) / 2, yp, (hi - lo) / 2, thick / 2))
    return np.array(bars, dtype=np.float64).reshape(-1, 4)


def _cell_walls(W, H, cell, thick):
    """Thin vertical and horizontal walls dividing [0, W] x [0, H] into cells of side ~cell."""
    bars = []
    mx = max(1, int(round(W / cell)))
    my = max(1, int(round(H / cell)))
    for j in range(1, mx):
        bars.append((j * W / mx, H / 2, thick / 2, H / 2))
    for j in range(1, my):
        bars.append((W / 2, j * H / my, W / 2, thick / 2))
    return np.array(bars, dtype=np.float64).reshape(-1, 4)


def settle(n, W, rng, mode="sieve", hole=1.25, pitch=None, cell=3.0, budget=30000,
           k=1000.0, dt=0.004, chunk=500, temp_shake=0.3, piston=True, init=None, gravity=(0.0, -1.0)):
    """Feed n squares into a container of width W, shake, then lower a lid to height W.

    Phases (budget = total integrator steps, identical across modes for a fair comparison):
      feed (45%): `mode` decides how the squares arrive (see module docstring), gravity down,
        open top, shaking at temp_shake;
      piston (40%): a lid comes down from the pile top to W while the box is tapped with a
        falling temperature (granular compaction under load);
      quench (15%): no kicks, gravity off, damped settle at side W.
    Returns the configuration (x, y, t radians); with piston=False the lid phase is skipped
    and the pile is just tapped under gravity. mode="seeded" skips the feed: `init` (x, y, t)
    is loosened by 3%, placed on the floor of the container and shaken from there.
    gravity=(-0.707, -0.707) tilts the container 45 degrees, so the pile settles into a corner.
    """
    pitch = pitch or hole + 0.6
    empty = np.zeros((0, 4))
    vx, vy, vt = np.zeros(n), np.zeros(n), np.zeros(n)

    def run(x, y, t, bars, steps, temp, H=BIG, g=1.0, damp=0.02):
        simulate(x, y, t, vx, vy, vt, W, H, bars, steps, k, gravity[0], gravity[1], g, temp, dt, damp,
                 int(rng.integers(1 << 30)))

    feed = int(budget * 0.45) // chunk
    press = int(budget * 0.40) // chunk
    quench = max(1, (budget // chunk) - feed - press)
    used = 0
    if mode == "seeded":
        x, y, t = (np.array(a, dtype=np.float64) for a in init)
        w = 0.5 * (np.abs(np.cos(t)) + np.abs(np.sin(t)))
        x, y = x - (x - w).min(), y - (y - w).min()
        span = max((x + w).max(), (y + w).max())
        x, y = x * (W / span), y * 1.03 * (W / span)
        for c in range(feed):
            run(x, y, t, empty, chunk, temp_shake)
            used += 1
    elif mode == "cells":
        # The walled region is a little taller than the pile will be; each cell fills from a
        # loose lattice of squares scattered over it, and the walls thin away during the feed.
        H0 = max(W, n / W * 1.15 + 1.0)
        walls = _cell_walls(W, H0, cell, 0.12)
        cols = max(1, int((W - 0.2) // 1.45))
        rows = int(np.ceil(n / cols))
        x, y, t = _hopper(n, W, 0.0, rng, spacing=W / cols)
        y = 0.6 + (y - y.min()) * ((H0 - 1.2) / max(1e-9, (rows - 1) * W / cols)) if rows > 1 else y + 0.6
        for c in range(feed):
            thin = max(0.0, 1.0 - 1.4 * c / feed)
            bars = walls.copy()
            vert = bars[:, 2] < bars[:, 3]
            bars[vert, 2] *= thin
            bars[~vert, 3] *= thin
            run(x, y, t, bars if thin > 0 else empty, chunk, temp_shake)
            used += 1
    else:
        yp = W + 1.0
        x, y, t = _hopper(n, W, yp + 1.0, rng)
        amp = pitch / 2
        for c in range(feed):
            bars = _plate(W, yp, hole, pitch, amp * np.sin(0.7 * c)) if mode == "sieve" else empty
            run(x, y, t, bars, chunk, temp_shake)
            used += 1
    w = 0.5 * (np.abs(np.cos(t)) + np.abs(np.sin(t)))
    top = float((y + w).max())
    for c in range(press):
        frac = (c + 1) / press
        H = top + (W - top) * min(1.0, 1.25 * frac) if piston else BIG
        temp = temp_shake * (1 - frac) ** 2
        run(x, y, t, empty, chunk // 2, temp, H=H)
        run(x, y, t, empty, chunk - chunk // 2, 0.0, H=H, damp=0.05)
    for c in range(quench):
        run(x, y, t, empty, chunk, 0.0, H=W if piston else BIG, g=0.0, damp=0.1)
    return x, y, t


def close_box(x, y, t):
    """Shift the configuration into the positive quadrant and close it into a feasible square box."""
    w = 0.5 * (np.abs(np.cos(t)) + np.abs(np.sin(t)))
    x = x - (x - w).min()
    y = y - (y - w).min()
    L = max((x + w).max(), (y + w).max())
    res = compact_squeeze(x, y, t, L)
    if res is None:
        return None
    x, y, t, L = res
    if not feasible(x, y, t, L):
        return None
    return x, y, t, float(L)
