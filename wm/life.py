"""Game-of-Life packing: squares that move by local neighbour rules while the box shrinks.

Every square sees only its neighbours (squares whose centers are within 2 of its own, plus a
wall it is close to) and updates its position and angle synchronously from what it sees. A rule
is a short vector of coefficients for a fixed set of local terms, each one an idea a rule
designer might have:

    rep    push apart in proportion to overlap depth (plain contact repulsion)
    coh    pull toward a neighbour across a small gap (surface tension; closes slack)
    align  turn toward the neighbour's angle mod 90 deg (negative: turn away, tilted domains)
    roll   overlap applied off-center makes a torque (rigid-body contact)
    slide  aligned neighbours slide along their shared face toward offset 0 (columns) or +-1/2
    space  soft center-to-center repulsion inside radius R (spreads holes evenly)
    axis   turn toward the box axes (negative: toward 45 deg)
    wall   pull toward a wall across a small gap (packs against the boundary)
    noise  random kicks that decay over the run (annealing)

plus the schedule: how far the box first melts out, how fast it shrinks while the worst overlap
is under a tolerance (and grows back when it is over), and momentum. The neighbourhood terms
are the same for every n, so one rule can be run on any puzzle size; that is the point: rules
are judged by the packings they reach across many n, not by one.

`run_rule` plays a rule from a seed packing (stored best, or a loose random layout) and returns
the end state, which may still overlap slightly; `play` adds the standard `squeeze` finish so
the result is exactly feasible and can be offered to the store.
"""

import math
from dataclasses import dataclass

import numpy as np
from numba import njit

from .pack_opt import MARGIN, feasible, squeeze

# Name, default, low, high. A genome is a vector in this order; `decode` clips it to the box.
PARAMS = [
    ("rep", 0.5, 0.0, 2.0),
    ("coh", 0.0, -0.2, 0.2),
    ("coh_range", 0.1, 0.01, 0.5),
    ("align", 0.0, -0.1, 0.1),
    ("roll", 0.0, -1.0, 1.0),
    ("slide", 0.0, -0.05, 0.05),
    ("space", 0.0, 0.0, 0.1),
    ("space_r", 1.2, 1.0, 1.6),
    ("axis", 0.0, -0.05, 0.05),
    ("wall", 0.0, -0.1, 0.1),
    ("noise", 0.0, 0.0, 0.05),
    ("noise_decay", 3.0, 0.0, 10.0),  # noise * exp(-noise_decay * step / steps)
    ("melt", 0.02, 0.0, 0.08),  # box starts (1 + melt) times the seed side
    ("shrink", 2e-4, 1e-5, 2e-3),  # relative box shrink per step while jammed-free
    ("tol", 0.02, 0.001, 0.1),  # worst overlap depth tolerated before the box grows back
    ("momentum", 0.5, 0.0, 0.95),
]
NAMES = [p[0] for p in PARAMS]
LOW = np.array([p[2] for p in PARAMS])
HIGH = np.array([p[3] for p in PARAMS])
DEFAULT = np.array([p[1] for p in PARAMS])


def decode(z) -> np.ndarray:
    return np.clip(np.asarray(z, dtype=np.float64), LOW, HIGH)


def as_dict(c) -> dict:
    return {k: float(v) for k, v in zip(NAMES, decode(c))}


def from_dict(d: dict) -> np.ndarray:
    c = DEFAULT.copy()
    for k, v in d.items():
        c[NAMES.index(k)] = v
    return c


@njit(cache=True)
def _h(a):
    return abs(np.cos(a)) + abs(np.sin(a))


@njit(cache=True)
def _cells(x, y, L, cell):
    """Bucket squares into a grid of `cell`-sized cells (linked lists head/nxt) for neighbour search."""
    m = max(1, int(L / cell) + 3)
    head = -np.ones(m * m, np.int64)
    nxt = -np.ones(x.shape[0], np.int64)
    for i in range(x.shape[0]):
        cx = min(m - 1, max(0, int(x[i] / cell) + 1))
        cy = min(m - 1, max(0, int(y[i] / cell) + 1))
        k = cy * m + cx
        nxt[i] = head[k]
        head[k] = i
    return head, nxt, m


@njit(cache=True)
def local_update(x, y, t, L, c, noise_amp, rng_kicks):
    """One synchronous step of the rule: per-square displacement (dx, dy, dt) and the worst overlap.

    Each square's move is a sum over its neighbours (and walls) of the rule terms, so it depends
    only on its own neighbourhood.
    """
    n = x.shape[0]
    rep, coh, coh_r, align, roll, slide, space, space_r, axis, wall = (
        c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9])
    fx = np.zeros(n)
    fy = np.zeros(n)
    ft = np.zeros(n)
    worst = 0.0
    co = np.cos(t)
    si = np.sin(t)
    reach = 2.0  # neighbourhood radius: touching squares are within sqrt 2, the rest is gap sensing
    head, nxt, m = _cells(x, y, L, reach)
    for i in range(n):
        cx = min(m - 1, max(0, int(x[i] / reach) + 1))
        cy = min(m - 1, max(0, int(y[i] / reach) + 1))
        for ox in range(-1, 2):
            for oy in range(-1, 2):
                gx_, gy_ = cx + ox, cy + oy
                if gx_ < 0 or gy_ < 0 or gx_ >= m or gy_ >= m:
                    continue
                j = head[gy_ * m + gx_]
                while j >= 0:
                    if j > i:
                        dx = x[j] - x[i]
                        dy = y[j] - y[i]
                        d2 = dx * dx + dy * dy
                        if d2 < reach * reach:
                            dist = math.sqrt(d2) + 1e-12
                            delta = t[j] - t[i]
                            rad = 0.5 + 0.5 * _h(delta) + MARGIN
                            best = 1e9
                            ax = 0.0
                            ay = 0.0
                            pb = 0.0
                            for k in range(4):
                                if k == 0:
                                    bx, by = co[i], si[i]
                                elif k == 1:
                                    bx, by = -si[i], co[i]
                                elif k == 2:
                                    bx, by = co[j], si[j]
                                else:
                                    bx, by = -si[j], co[j]
                                proj = dx * bx + dy * by
                                o = rad - abs(proj)
                                if o < best:
                                    best, ax, ay, pb = o, bx, by, proj
                            sg = 1.0 if pb >= 0 else -1.0
                            nx, ny = sg * ax, sg * ay  # unit normal from i toward j
                            tx, ty = -ny, nx
                            q = dx * tx + dy * ty  # offset of j along the shared face
                            if best > worst:
                                worst = best
                            fn = 0.0  # force on j along n (i gets the opposite)
                            ftan = 0.0
                            tau = 0.0  # torque on i (j gets the opposite)
                            if best > 0:
                                fn += rep * best
                                tau += roll * best * q
                                prox = 1.0
                            else:
                                prox = math.exp(best / coh_r)
                                fn -= coh * prox
                            if prox > 1e-3:
                                s4 = math.sin(4.0 * delta)
                                tau += align * s4 * prox
                                aligned = 0.5 * (1.0 + math.cos(4.0 * delta))
                                ftan += slide * math.sin(2.0 * math.pi * q) * aligned * prox
                            if dist < space_r:
                                sp = space * (space_r - dist)
                                fx[j] += sp * dx / dist
                                fy[j] += sp * dy / dist
                                fx[i] -= sp * dx / dist
                                fy[i] -= sp * dy / dist
                            fx[j] += fn * nx + ftan * tx
                            fy[j] += fn * ny + ftan * ty
                            fx[i] -= fn * nx + ftan * tx
                            fy[i] -= fn * ny + ftan * ty
                            ft[i] += tau
                            ft[j] -= tau
                    j = nxt[j]
    for i in range(n):
        w = 0.5 * _h(t[i])
        ft[i] -= axis * math.sin(4.0 * t[i])
        for coord in range(2):
            p = x[i] if coord == 0 else y[i]
            lo = w - p  # > 0: through the low wall
            hi = p + w - L
            f = 0.0
            if lo > 0:
                f += rep * lo
                if lo > worst:
                    worst = lo
            elif lo > -0.5:
                f -= wall * math.exp(lo / coh_r)
            if hi > 0:
                f -= rep * hi
                if hi > worst:
                    worst = hi
            elif hi > -0.5:
                f += wall * math.exp(hi / coh_r)
            if coord == 0:
                fx[i] += f
            else:
                fy[i] += f
    if noise_amp > 0:
        fx += noise_amp * rng_kicks[0]
        fy += noise_amp * rng_kicks[1]
        ft += noise_amp * rng_kicks[2]
    return fx, fy, ft, worst


@njit(cache=True)
def _run(x, y, t, L, c, steps, seed):
    np.random.seed(seed)
    n = x.shape[0]
    noise, decay, melt, shrink, tol, mom = c[10], c[11], c[12], c[13], c[14], c[15]
    x, y = x * (1 + melt), y * (1 + melt)
    L = L * (1 + melt)
    t = t.copy()
    vx, vy, vt = np.zeros(n), np.zeros(n), np.zeros(n)
    cap = 0.05  # largest move per step, keeps any rule numerically stable
    for s in range(steps):
        amp = noise * math.exp(-decay * s / steps)
        kicks = np.random.standard_normal((3, n))
        fx, fy, ft, worst = local_update(x, y, t, L, c, amp, kicks)
        vx = mom * vx + fx
        vy = mom * vy + fy
        vt = mom * vt + ft
        for i in range(n):
            vx[i] = min(cap, max(-cap, vx[i]))
            vy[i] = min(cap, max(-cap, vy[i]))
            vt[i] = min(cap, max(-cap, vt[i]))
        x += vx
        y += vy
        t += vt
        # Box pressure: the walls move in only while every contact is nearly resolved.
        k = 1 - shrink if worst < tol else 1 + 0.5 * shrink
        x *= k
        y *= k
        L *= k
    return x, y, t, L


def run_rule(c, x, y, t, L, steps=600, seed=0):
    """Play rule c from a packing (radians) in [0, L]^2; returns the end state (may overlap a little)."""
    x, y, t = (np.asarray(a, dtype=np.float64) for a in (x, y, t))
    return _run(x, y, t, float(L), decode(c), int(steps), int(seed))


@dataclass
class Played:
    x: np.ndarray
    y: np.ndarray
    t: np.ndarray
    L: float  # feasible side after the squeeze finish (inf if squeeze failed)
    rule_L: float  # box side the rule itself reached


def play(c, x, y, t, L, steps=600, seed=0, finish_rounds=30) -> Played:
    """run_rule, then squeeze to an exactly feasible packing."""
    rx, ry, rt, rL = run_rule(c, x, y, t, L, steps, seed)
    r = squeeze(rx, ry, rt, rL, rounds=finish_rounds, step=0.001)
    if r is None or not feasible(*r):
        return Played(rx, ry, rt, math.inf, rL)
    return Played(r[0], r[1], r[2], float(r[3]), rL)


def random_seed(n: int, rng, loose: float = 1.3):
    """Loose random layout: centers uniform in a box loose * sqrt(n) wide, random angles."""
    L = loose * math.sqrt(n) + 1
    return (rng.uniform(0.7, L - 0.7, n), rng.uniform(0.7, L - 0.7, n), rng.uniform(-np.pi / 4, np.pi / 4, n), L)
