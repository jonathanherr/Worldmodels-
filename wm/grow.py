"""Asymmetric growth packer: seed tiny squares in a box, grow them unevenly until they jam.

Squares i have their own side s_i (centers x, y, angle t in radians) inside the box [0, 1]^2.
Each growth step picks a set G of squares, grows them, and relaxes the overlaps they cause.
The knobs this module varies:

  simultaneity  how many squares grow per step: k = 1 (one at a time), a group, or all n,
                and how the group is picked (random, the smallest first, a spatial cluster).
  direction     iso (grow about the center), side (one face stays put, the opposite face moves:
                a fixed random side per square), free (towards the face with most room right now),
                corner (two faces move, the opposite corner stays put).
  rate          equal, random (a fixed lognormal rate per square), contacts (1 / (1 + touching
                neighbours): open squares grow faster), localmin (a square grows only while it is
                no bigger than its smallest neighbour times 1 + band).
  adjacency     push (every square may move during relaxation), local (only growing squares and
                the squares they touch), nopush (only the growing squares move; growth into free
                space only), independent (push, but no two touching squares grow in the same step).

A square whose growth fails repeatedly is jammed. A cap s_i <= min(s) * (1 + band) keeps sizes
from running away (band = inf disables it). When every square is jammed or capped the run ends;
shrinking every square to min(s) about its center keeps the packing overlap-free, so the jammed
state is a unit-square packing in a box of side 1 / min(s), which `finish` then squeezes.
"""

from dataclasses import dataclass

import numpy as np
from numba import njit


@njit(cache=True)
def _h(a):
    return abs(np.cos(a)) + abs(np.sin(a))


@njit(cache=True)
def _dh(a):
    return -np.sign(np.cos(a)) * np.sin(a) + np.sign(np.sin(a)) * np.cos(a)


@njit(cache=True)
def energy_grad(x, y, t, s, L):
    """Overlap energy of squares with sides s in [0, L]^2, its gradient, and the worst pair depth."""
    n = x.shape[0]
    gx = np.zeros(n)
    gy = np.zeros(n)
    gt = np.zeros(n)
    e = 0.0
    worst = 0.0
    smax = 0.0
    for i in range(n):
        if s[i] > smax:
            smax = s[i]
    reach = 2.0 * smax * smax + 1e-12
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[j] - x[i]
            dy = y[j] - y[i]
            if dx * dx + dy * dy >= reach:
                continue
            r = 0.5 * (s[i] + s[j]) * 1.41421357
            if dx * dx + dy * dy >= r * r:
                continue
            delta = t[j] - t[i]
            hd = _h(delta)
            best = 1e9
            kb = 0
            pb = 0.0
            for k in range(4):
                if k < 2:
                    phi = t[i] + 0.5 * np.pi * k
                    base = 0.5 * s[i] + 0.5 * s[j] * hd
                else:
                    phi = t[j] + 0.5 * np.pi * (k - 2)
                    base = 0.5 * s[j] + 0.5 * s[i] * hd
                p = dx * np.cos(phi) + dy * np.sin(phi)
                ov = base - abs(p)
                if ov < best:
                    best = ov
                    kb = k
                    pb = p
            if best <= 0.0:
                continue
            if best > worst:
                worst = best
            e += best * best
            g = 2.0 * best
            if kb < 2:
                phi = t[i] + 0.5 * np.pi * kb
            else:
                phi = t[j] + 0.5 * np.pi * (kb - 2)
            ax = np.cos(phi)
            ay = np.sin(phi)
            sg = 1.0 if pb >= 0 else -1.0
            # d best / d dx = -sg * ax ; dx = x_j - x_i
            gx[j] += g * (-sg * ax)
            gx[i] -= g * (-sg * ax)
            gy[j] += g * (-sg * ay)
            gy[i] -= g * (-sg * ay)
            # rotation of the normal's owner changes p
            dp = -dx * ay + dy * ax
            if kb < 2:
                gt[i] += g * (-sg * dp)
                other = s[j]
            else:
                gt[j] += g * (-sg * dp)
                other = s[i]
            dhd = _dh(delta) * 0.5 * other
            gt[j] += g * dhd
            gt[i] -= g * dhd
    for i in range(n):
        w = 0.5 * s[i] * _h(t[i])
        dw = 0.5 * s[i] * _dh(t[i])
        for c, which in ((x[i], 0), (y[i], 1)):
            lo = w - c
            hi = c + w - L
            if lo > 0:
                e += lo * lo
                if which == 0:
                    gx[i] -= 2 * lo
                else:
                    gy[i] -= 2 * lo
                gt[i] += 2 * lo * dw
                if lo > worst:
                    worst = lo
            if hi > 0:
                e += hi * hi
                if which == 0:
                    gx[i] += 2 * hi
                else:
                    gy[i] += 2 * hi
                gt[i] += 2 * hi * dw
                if hi > worst:
                    worst = hi
    return e, gx, gy, gt, worst


@njit(cache=True)
def relax(x, y, t, s, L, mov, maxiter, tol, rotw=0.3):
    """FIRE minimization of the overlap energy, moving only squares with mov[i] = True.

    Stops when the worst depth is below tol. Returns (x, y, t, worst)."""
    n = x.shape[0]
    x = x.copy()
    y = y.copy()
    t = t.copy()
    vx = np.zeros(n)
    vy = np.zeros(n)
    vt = np.zeros(n)
    smean = 0.0
    for i in range(n):
        smean += s[i]
    smean /= n
    dt = 0.1
    dtmax = 0.5
    alpha = 0.1
    npos = 0
    worst = 0.0
    rot = rotw  # weight of the angle coordinates in the dynamics
    for it in range(maxiter):
        e, gx, gy, gt, worst = energy_grad(x, y, t, s, L)
        if worst <= tol:
            break
        for i in range(n):
            if not mov[i]:
                gx[i] = 0.0
                gy[i] = 0.0
                gt[i] = 0.0
                vx[i] = 0.0
                vy[i] = 0.0
                vt[i] = 0.0
            else:
                gt[i] *= rot
        P = 0.0
        vn = 0.0
        fn = 0.0
        for i in range(n):
            P -= vx[i] * gx[i] + vy[i] * gy[i] + vt[i] * gt[i]
            vn += vx[i] * vx[i] + vy[i] * vy[i] + vt[i] * vt[i]
            fn += gx[i] * gx[i] + gy[i] * gy[i] + gt[i] * gt[i]
        vn = np.sqrt(vn)
        fn = np.sqrt(fn) + 1e-300
        if P > 0:
            for i in range(n):
                vx[i] = (1 - alpha) * vx[i] - alpha * gx[i] / fn * vn
                vy[i] = (1 - alpha) * vy[i] - alpha * gy[i] / fn * vn
                vt[i] = (1 - alpha) * vt[i] - alpha * gt[i] / fn * vn
            npos += 1
            if npos > 5:
                dt = min(dt * 1.1, dtmax)
                alpha *= 0.99
        else:
            npos = 0
            dt *= 0.5
            alpha = 0.1
            for i in range(n):
                vx[i] = 0.0
                vy[i] = 0.0
                vt[i] = 0.0
        for i in range(n):
            vx[i] -= dt * gx[i]
            vy[i] -= dt * gy[i]
            vt[i] -= dt * gt[i]
            x[i] += dt * vx[i]
            y[i] += dt * vy[i]
            t[i] += dt * vt[i]
    return x, y, t, worst


@njit(cache=True)
def contacts(x, y, t, s, gap):
    """Boolean adjacency: squares closer than gap (as separating-axis clearance)."""
    n = x.shape[0]
    A = np.zeros((n, n), dtype=np.bool_)
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[j] - x[i]
            dy = y[j] - y[i]
            r = 0.5 * (s[i] + s[j]) * 1.41421357 + gap
            if dx * dx + dy * dy >= r * r:
                continue
            hd = _h(t[j] - t[i])
            best = 1e9
            for k in range(4):
                if k < 2:
                    phi = t[i] + 0.5 * np.pi * k
                    base = 0.5 * s[i] + 0.5 * s[j] * hd
                else:
                    phi = t[j] + 0.5 * np.pi * (k - 2)
                    base = 0.5 * s[j] + 0.5 * s[i] * hd
                ov = base - abs(dx * np.cos(phi) + dy * np.sin(phi))
                if ov < best:
                    best = ov
            if best > -gap:
                A[i, j] = True
                A[j, i] = True
    return A


@dataclass
class GrowConfig:
    k: int = 0               # squares growing per step; 0 = all
    pick: str = "random"     # random | smallest | cluster | void (squares facing the biggest gaps)
    direction: str = "iso"   # iso | side | free | corner
    rate: str = "equal"      # equal | random | contacts | localmin
    adjacency: str = "push"  # push | local | nopush | independent
    band: float = 0.05       # cap s_i <= min(s) * (1 + band)
    angles: str = "random"   # random | zero (seed angles)
    g0: float = 0.05         # initial relative growth per step
    gmin: float = 2e-4       # relative growth below which a square counts as jammed
    max_steps: int = 20000
    revivals: int = 3        # times jammed squares get their growth step back when all are stuck
    relax_iters: int = 1000
    rotw: float = 0.3        # angle weight in the relaxation dynamics
    tol: float = 1e-6        # accepted residual depth, relative to min(s)
    seed: int = 0
    s0: float = 0.0          # seed side (0 = 0.15 / sqrt(n)); with max_steps=0 this is a no-growth control
    record: int = 0          # keep a snapshot every `record` accepted steps (0 = off)


def _wall_room(x, y, t, s, i, others):
    """Room in front of each of square i's four faces (+u, +v, -u, -v): distance to the nearest
    other center or wall along that face normal, a cheap proxy for free space."""
    room = np.zeros(4)
    for k in range(4):
        phi = t[i] + 0.5 * np.pi * k
        ax, ay = np.cos(phi), np.sin(phi)
        # walls
        lim = []
        for c, a in ((x[i], ax), (y[i], ay)):
            if a > 1e-9:
                lim.append((1.0 - c) / a)
            elif a < -1e-9:
                lim.append(-c / a)
        best = min(lim) if lim else 1e9
        dx = x[others] - x[i]
        dy = y[others] - y[i]
        p = dx * ax + dy * ay
        q = np.abs(-dx * ay + dy * ax)
        front = (p > 0) & (q < 0.5 * (s[i] + s[others]) * 1.2)
        if front.any():
            best = min(best, float(p[front].min()))
        room[k] = best
    return room


def grow(n: int, cfg: GrowConfig):
    """Run one growth simulation. Returns dict with x, y, t, s (box [0,1]^2) and stats."""
    rng = np.random.default_rng(cfg.seed)
    s0 = cfg.s0 if cfg.s0 > 0 else 0.15 / np.sqrt(n)
    x = rng.uniform(s0, 1 - s0, n)
    y = rng.uniform(s0, 1 - s0, n)
    t = rng.uniform(0, np.pi / 2, n) if cfg.angles == "random" else np.zeros(n)
    s = np.full(n, s0)
    x, y, t, _ = relax(x, y, t, s, 1.0, np.ones(n, np.bool_), 2000, 1e-12)
    g = np.full(n, cfg.g0)
    rate = rng.lognormal(0.0, 0.5, n) if cfg.rate == "random" else np.ones(n)
    side = rng.integers(0, 4, n)  # fixed face for direction="side"
    corner = rng.integers(0, 4, n)
    k = n if cfg.k <= 0 else min(cfg.k, n)
    steps = fails = revivals = accepted = 0
    frames = []
    while steps < cfg.max_steps:
        steps += 1
        smin = s.min()
        cap = smin * (1 + cfg.band)
        A = None
        if cfg.rate in ("contacts", "localmin") or cfg.adjacency in ("local", "independent") or cfg.pick == "cluster":
            A = contacts(x, y, t, s, 0.02 * smin)
        elig = (g >= cfg.gmin) & (s < cap)
        if cfg.rate == "localmin":
            nb_min = np.where(A, s[None, :], np.inf).min(axis=1)
            elig &= s <= nb_min * (1 + cfg.band)
        idx = np.flatnonzero(elig)
        if idx.size == 0:
            stuck = np.flatnonzero((g < cfg.gmin) & (s < cap))
            if revivals >= cfg.revivals or stuck.size == 0:
                break
            revivals += 1
            g[stuck] = cfg.g0
            continue
        if k >= idx.size:
            G = idx
        elif cfg.pick == "smallest":
            G = idx[np.argsort(s[idx] + 1e-9 * rng.random(idx.size))[:k]]
        elif cfg.pick == "void":  # squares facing the biggest gaps grow first
            room = np.array([_wall_room(x, y, t, s, i, np.flatnonzero(np.arange(n) != i)).max() - 0.5 * s[i]
                             for i in idx])
            G = idx[np.argsort(-room + 1e-9 * rng.random(idx.size))[:k]]
        elif cfg.pick == "cluster":
            c = rng.choice(idx)
            d = (x[idx] - x[c]) ** 2 + (y[idx] - y[c]) ** 2
            G = idx[np.argsort(d)[:k]]
        else:
            G = rng.choice(idx, size=k, replace=False)
        if cfg.adjacency == "independent" and G.size > 1:
            keep = []
            for i in rng.permutation(G):
                if not any(A[i, j] for j in keep):
                    keep.append(i)
            G = np.array(keep)
        ds = g[G] * s[G] * rate[G]
        if cfg.rate == "contacts":
            ds = ds * 2.0 / (1.0 + A[G].sum(axis=1))
        ds = np.minimum(ds, np.maximum(cap - s[G], 0.0) + 1e-15)
        old = (x.copy(), y.copy(), t.copy(), s.copy())
        s[G] += ds
        if cfg.direction != "iso":
            for gi, i in enumerate(G):
                if cfg.direction == "side":
                    faces = [side[i]]
                elif cfg.direction == "corner":
                    faces = [corner[i], (corner[i] + 1) % 4]
                else:  # free: the face with most room
                    others = np.flatnonzero(np.arange(n) != i)
                    faces = [int(np.argmax(_wall_room(x, y, t, s, i, others)))]
                for f in faces:
                    phi = t[i] + 0.5 * np.pi * f
                    x[i] += 0.5 * ds[gi] * np.cos(phi)
                    y[i] += 0.5 * ds[gi] * np.sin(phi)
        if cfg.adjacency == "nopush":
            mov = np.zeros(n, np.bool_)
            mov[G] = True
        elif cfg.adjacency == "local":
            mov = np.zeros(n, np.bool_)
            mov[G] = True
            mov |= A[G].any(axis=0)
        else:
            mov = np.ones(n, np.bool_)
        nx, ny, nt, worst = relax(x, y, t, s, 1.0, mov, cfg.relax_iters, cfg.tol * smin, cfg.rotw)
        if worst <= cfg.tol * smin:
            x, y, t = nx, ny, nt
            g[G] = np.minimum(g[G] * 1.2, cfg.g0)
            accepted += 1
            if cfg.record and accepted % cfg.record == 0:
                frames.append((x.copy(), y.copy(), t.copy(), s.copy(), G.copy()))
        else:
            fails += 1
            _, gx, gy, gt, _ = energy_grad(nx, ny, nt, s, 1.0)
            bad = (np.abs(gx) + np.abs(gy) + np.abs(gt)) > 0
            hit = G[bad[G]]
            g[hit if hit.size else G] *= 0.5
            x, y, t, s = old
    return {"x": x, "y": y, "t": t, "s": s, "steps": steps, "fails": fails, "revivals": revivals,
            "side_direct": float(1.0 / s.min()), "frames": frames}


def to_unit(res):
    """Shrink every square to min(s) and rescale so squares are unit: (x, y, t, L)."""
    m = res["s"].min()
    return res["x"] / m, res["y"] / m, res["t"], 1.0 / m


def finish(res, mode="squeeze"):
    """Turn a jammed state into an exactly feasible unit packing. mode: squeeze | compact."""
    from .pack_opt import compact_squeeze, squeeze

    x, y, t, L = to_unit(res)
    if mode == "compact":
        return compact_squeeze(x, y, t, L)
    return squeeze(x, y, t, L * 1.000001, rounds=30, step=0.002, method="fire", maxiter=3000)
