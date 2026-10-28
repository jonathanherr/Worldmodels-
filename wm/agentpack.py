"""One agent per square: a cheap multi-agent packer that starts from scratch.

Every unit square is an agent. Its local goals are to overlap nobody, to sit inside the box and
to keep contacts; the shared global goal is a smaller box. The box shrinks whenever all agents
are content (zero overlap energy) and the agents get a few rounds to make room when it does not.

Gradient relaxation (FIRE on `pack_opt.energy_grad`) is every agent following its local goal at
once. The point of this module is the three non-local behaviours on top of it, which differ from
the earlier neighbour rules (wm/life.py), shaking (sieve) and perturbation of record packings:

* hop      a stressed agent looks over the whole box for a hole (lowest insertion energy over
           sampled positions and the angles its neighbours use) and jumps there if it is less
           crowded than where it sits.
* slide    negotiation along a corridor: a stressed agent asks the agents between it and the
           nearest hole to shift one step towards the hole, so a vacancy travels to the crowd
           like a tile in a sliding puzzle. Accepted only if the total energy drops.
* coalition agents with the same tilt that touch edge to edge form a coalition (a row, a strip
           or a block) and move as one rigid body: a small rotation or translation of the whole
           group, then a short relax, accepted Metropolis-style.

`agent_pack(n, ...)` runs the whole thing from a random gas and returns the smallest feasible
packing it reached, after a final `compact_squeeze`.
"""

import numpy as np
from numba import njit

from .pack_opt import MARGIN, compact_squeeze, energy_grad, feasible, relax_fire


@njit(cache=True)
def _pair_depth(xi, yi, ti, xj, yj, tj, margin):
    """Separating-axis overlap depth of two unit squares (<= 0 means apart)."""
    dx = xj - xi
    dy = yj - yi
    if dx * dx + dy * dy >= 2.0 + 4 * margin + 1e-9:
        return -1.0
    delta = tj - ti
    rad = 0.5 + 0.5 * (abs(np.cos(delta)) + abs(np.sin(delta))) + margin
    best = 1e9
    for k in range(4):
        if k == 0:
            ax, ay = np.cos(ti), np.sin(ti)
        elif k == 1:
            ax, ay = -np.sin(ti), np.cos(ti)
        elif k == 2:
            ax, ay = np.cos(tj), np.sin(tj)
        else:
            ax, ay = -np.sin(tj), np.cos(tj)
        o = rad - abs(dx * ax + dy * ay)
        if o < best:
            best = o
    return best


@njit(cache=True)
def _wall(px, py, pt, L):
    w = 0.5 * (abs(np.cos(pt)) + abs(np.sin(pt)))
    e = 0.0
    for p in (px, py):
        lo = w - p
        hi = p + w - L
        if lo > 0:
            e += lo * lo
        if hi > 0:
            e += hi * hi
    return e


@njit(cache=True)
def stress(x, y, t, L, margin):
    """Each agent's own overlap energy: half of every pair it is in plus its wall penetration."""
    n = x.shape[0]
    s = np.zeros(n)
    for i in range(n):
        s[i] += _wall(x[i], y[i], t[i], L)
        for j in range(i + 1, n):
            d = _pair_depth(x[i], y[i], t[i], x[j], y[j], t[j], margin)
            if d > 0:
                s[i] += 0.5 * d * d
                s[j] += 0.5 * d * d
    return s


@njit(cache=True)
def insertion_energy(x, y, t, skip, px, py, pt, L, margin):
    """Energy a square at (px, py, pt) would have against everyone except agent `skip`."""
    e = _wall(px, py, pt, L)
    for j in range(x.shape[0]):
        if j == skip:
            continue
        d = _pair_depth(px, py, pt, x[j], y[j], t[j], margin)
        if d > 0:
            e += d * d
    return e


@njit(cache=True)
def best_hole(x, y, t, skip, L, cand_x, cand_y, angles, margin):
    """The lowest insertion energy over candidate positions x angles. Returns (e, px, py, pt)."""
    be, bx, by, bt = 1e18, 0.0, 0.0, 0.0
    for c in range(cand_x.shape[0]):
        for a in range(angles.shape[0]):
            e = insertion_energy(x, y, t, skip, cand_x[c], cand_y[c], angles[a], L, margin)
            if e < be:
                be, bx, by, bt = e, cand_x[c], cand_y[c], angles[a]
    return be, bx, by, bt


def _energy(x, y, t, L, margin=MARGIN):
    return energy_grad(x, y, t, L, margin)[0]


def _tilt(t):
    """Angle modulo the square's quarter-turn symmetry, in [0, pi/2)."""
    return np.mod(t, np.pi / 2)


def coalitions(x, y, t, tol=0.03, reach=1.15):
    """Groups of agents with the same tilt whose centres are about one side apart (edge contact).

    Union-find over close pairs; returns a list of index arrays, largest first.
    """
    n = len(x)
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    tt = _tilt(t)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.hypot(dx, dy)
    dtilt = np.abs(tt[:, None] - tt[None, :])
    dtilt = np.minimum(dtilt, np.pi / 2 - dtilt)
    ii, jj = np.nonzero(np.triu((dist < reach) & (dtilt < tol), 1))
    for i, j in zip(ii, jj):
        a, b = find(i), find(j)
        if a != b:
            parent[a] = b
    roots = np.array([find(i) for i in range(n)])
    groups = [np.nonzero(roots == r)[0] for r in np.unique(roots)]
    return sorted(groups, key=len, reverse=True)


class AgentPacker:
    """State and behaviours of the square agents for one n."""

    def __init__(self, n, rng, margin=MARGIN, relax_iters=400, holes=300):
        self.n, self.rng, self.margin = n, rng, margin
        self.relax_iters, self.holes = relax_iters, holes
        self.stats = {"hop": 0, "slide": 0, "coalition": 0, "tries": 0}

    # -- the agents' shared local behaviour -------------------------------------------------
    def relax(self, x, y, t, L, iters=None):
        x, y, t, e = relax_fire(x, y, t, L, self.margin, iters or self.relax_iters)
        return x, y, t, e

    # -- hop: jump to the emptiest place in the box -----------------------------------------
    def hop(self, x, y, t, L, i, s):
        cx = self.rng.uniform(0.5, L - 0.5, self.holes)
        cy = self.rng.uniform(0.5, L - 0.5, self.holes)
        # Angles worth trying: axis, 45 degrees, own, and the tilts of a few random agents.
        others = t[self.rng.integers(0, self.n, 4)]
        angles = np.concatenate([[0.0, np.pi / 4, t[i]], others])
        e, px, py, pt = best_hole(x, y, t, i, L, cx, cy, angles, self.margin)
        # Refine the hole locally: small jitter around the winner.
        jx = px + self.rng.normal(0, 0.15, 60)
        jy = py + self.rng.normal(0, 0.15, 60)
        jt = pt + np.array([0.0, 0.05, -0.05])
        e2, px2, py2, pt2 = best_hole(x, y, t, i, L, jx, jy, jt, self.margin)
        if e2 < e:
            e, px, py, pt = e2, px2, py2, pt2
        here = insertion_energy(x, y, t, i, x[i], y[i], t[i], L, self.margin)
        if e < 0.5 * here:
            x, y, t = x.copy(), y.copy(), t.copy()
            x[i], y[i], t[i] = px, py, pt
            return x, y, t, True
        return x, y, t, False

    # -- slide: agents in the corridor to the nearest hole shift towards it ----------------
    def slide(self, x, y, t, L, i, e_now):
        cx = self.rng.uniform(0.5, L - 0.5, self.holes)
        cy = self.rng.uniform(0.5, L - 0.5, self.holes)
        ins = np.array([insertion_energy(x, y, t, -1, cx[c], cy[c], t[i], L, self.margin)
                        for c in range(self.holes)])
        # Prefer near, empty holes: insertion energy plus a distance cost.
        d = np.hypot(cx - x[i], cy - y[i])
        k = int(np.argmin(ins + 0.02 * d))
        hx, hy = cx[k], cy[k]
        v = np.array([hx - x[i], hy - y[i]])
        dist = np.linalg.norm(v)
        if dist < 0.3:
            return x, y, t, False
        u = v / dist
        # Corridor: agents whose centre projects onto [0, dist] and lies within ~0.7 of the line.
        px = x - x[i]
        py = y - y[i]
        along = px * u[0] + py * u[1]
        across = np.abs(-px * u[1] + py * u[0])
        member = (along > -0.5) & (along < dist + 0.2) & (across < 0.7)
        if member.sum() < 1:
            return x, y, t, False
        best = None
        for step in (0.5, 0.25, 0.12):
            nx, ny = x.copy(), y.copy()
            # Agents further along shift less, so the vacancy is taken up gradually.
            f = step * np.clip(1.0 - along[member] / (dist + 1e-9) * 0.5, 0.3, 1.0)
            nx[member] += f * u[0]
            ny[member] += f * u[1]
            rx, ry, rt, e = self.relax(nx, ny, t, L, iters=150)
            if e < e_now and (best is None or e < best[3]):
                best = (rx, ry, rt, e)
        if best is not None:
            return best[0], best[1], best[2], True
        return x, y, t, False

    # -- coalition: a group with a shared tilt moves as one rigid body ---------------------
    def coalition_move(self, x, y, t, L, e_now, temp):
        groups = [g for g in coalitions(x, y, t) if len(g) >= 2]
        if not groups:
            return x, y, t, False
        g = groups[self.rng.integers(0, min(len(groups), 6))]
        cx, cy = x[g].mean(), y[g].mean()
        kind = self.rng.integers(0, 3)
        nx, ny, nt = x.copy(), y.copy(), t.copy()
        if kind == 0:  # rotate the group about its centroid
            a = self.rng.normal(0, 0.08)
            ca, sa = np.cos(a), np.sin(a)
            rx, ry = x[g] - cx, y[g] - cy
            nx[g] = cx + ca * rx - sa * ry
            ny[g] = cy + sa * rx + ca * ry
            nt[g] = t[g] + a
        elif kind == 1:  # translate the group
            d = self.rng.normal(0, 0.25, 2)
            nx[g] += d[0]
            ny[g] += d[1]
        else:  # turn the group to 45 degrees or back to axis: a strip forming or dissolving
            target = np.pi / 4 if abs(_tilt(t[g]).mean() - np.pi / 4) > 0.3 else 0.0
            a = target - _tilt(t[g]).mean()
            ca, sa = np.cos(a), np.sin(a)
            rx, ry = x[g] - cx, y[g] - cy
            nx[g] = cx + ca * rx - sa * ry
            ny[g] = cy + sa * rx + ca * ry
            nt[g] = t[g] + a
        nx = np.clip(nx, 0.5, L - 0.5)
        ny = np.clip(ny, 0.5, L - 0.5)
        rx, ry, rt, e = self.relax(nx, ny, nt, L, iters=200)
        if e < e_now or self.rng.random() < np.exp(-(e - e_now) / max(temp, 1e-12)):
            return rx, ry, rt, True
        return x, y, t, False

    # -- one negotiation round at a fixed box ------------------------------------------------
    def negotiate(self, x, y, t, L, rounds, use=("hop", "slide", "coalition")):
        x, y, t, e = self.relax(x, y, t, L)
        for r in range(rounds):
            if e == 0.0:
                return x, y, t, e
            s = stress(x, y, t, L, self.margin)
            order = np.argsort(-s)
            temp = 0.3 * e * (1 - r / rounds)
            for i in order[:3]:
                if s[i] <= 0:
                    break
                self.stats["tries"] += 1
                if "hop" in use:
                    nx, ny, nt, ok = self.hop(x, y, t, L, i, s)
                    if ok:
                        nx, ny, nt, ne = self.relax(nx, ny, nt, L)
                        if ne < e:
                            x, y, t, e = nx, ny, nt, ne
                            self.stats["hop"] += 1
                            continue
                if "slide" in use:
                    nx, ny, nt, ok = self.slide(x, y, t, L, i, e)
                    if ok:
                        nx, ny, nt, ne = self.relax(nx, ny, nt, L)
                        if ne < e:
                            x, y, t, e = nx, ny, nt, ne
                            self.stats["slide"] += 1
            if "coalition" in use and e > 0:
                nx, ny, nt, ok = self.coalition_move(x, y, t, L, e, temp)
                if ok:
                    nx, ny, nt, ne = self.relax(nx, ny, nt, L)
                    x, y, t, e = nx, ny, nt, ne
                    self.stats["coalition"] += 1
        return x, y, t, e


def random_gas(n, rng, density=0.45):
    L = np.sqrt(n / density)
    x = rng.uniform(0.75, L - 0.75, n)
    y = rng.uniform(0.75, L - 0.75, n)
    t = rng.uniform(0, np.pi / 2, n)
    return x, y, t, L


def agent_pack(n, seed=0, rounds=12, shrink=0.02, min_shrink=2e-4, max_fail=40,
               use=("hop", "slide", "coalition"), finish=True, verbose=False):
    """Pack n squares from a random gas with the agent behaviours. Returns (x, y, t, L, stats)."""
    rng = np.random.default_rng(seed)
    ag = AgentPacker(n, rng)
    x, y, t, L = random_gas(n, rng)
    # Loose start: relax until feasible.
    x, y, t, e = ag.relax(x, y, t, L, iters=3000)
    while not feasible(x, y, t, L):
        x, y, L = x * 1.05, y * 1.05, L * 1.05
        x, y, t, e = ag.relax(x, y, t, L, iters=3000)
    best = (x.copy(), y.copy(), t.copy(), L)
    step, fails = shrink, 0
    while step > min_shrink and fails < max_fail:
        bx, by, bt, bL = best
        Lt = bL * (1 - step)
        k = Lt / bL
        x, y, t, e = ag.negotiate(bx * k, by * k, bt.copy(), Lt, rounds, use)
        if e == 0.0 and feasible(x, y, t, Lt):
            best = (x, y, t, Lt)
            step = min(step * 1.3, shrink)
            if verbose:
                print(f"  L={Lt:.4f}  step={step:.4f}  {ag.stats}")
        else:
            step *= 0.6
            fails += 1
    x, y, t, L = best
    if finish:
        out = compact_squeeze(x, y, t, L)
        if out is not None and out[3] < L:
            x, y, t, L = out
    return x, y, t, L, ag.stats


def control_pack(n, seed=0, tries=1):
    """Baseline with the same start and no agent behaviours: random gas -> compact_squeeze."""
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(tries):
        x, y, t, L = random_gas(n, rng)
        out = compact_squeeze(x, y, t, L)
        if out is not None and (best is None or out[3] < best[3]):
            best = out
    return best
