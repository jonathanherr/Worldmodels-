"""Sequential placement policy for Squares in Squares: an evolved net places one square at a time.

The one-shot nets in `wm.neuroevo` and `wm.neat` map every square to a pose at once, and most of
their layouts squeeze back to the grid. Here the net is a policy instead. Given a target box of
side L and the squares placed so far, it scores every candidate spot (a fine grid of centers x a
few angles) from local features of that spot:

    x/L, y/L, distance to the corner, gap to the nearest wall, wall contacts, angle (as theta/45),
    gap to the nearest placed square, contacts with placed squares, local density, angle match
    with the nearest neighbour, fraction of squares already placed

and the best non-overlapping spot gets the next square (a spot may overlap by up to `tol`; the
squeeze fixes that). When nothing fits, the square goes where it overlaps least. After all n are
placed, `compact_squeeze` turns the layout into a tight valid packing, and its side is the
fitness. Fitness is smoother than for one-shot placement: the leftover overlap at placement time
is added as a small tie-break, so layouts that nearly fit beat ones that clearly do not.

CMA-ES evolves the weights of a small MLP (11 -> hidden -> 1), as in `wm.neuroevo`.
"""

import math
import multiprocessing as mp
import os

import numpy as np
from numba import njit

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

from .pack_opt import compact, squeeze  # noqa: E402

N_FEAT = 11
CONTACT = 0.05  # a gap below this counts as touching


@njit(cache=True)
def _pair_depth(x1, y1, t1, x2, y2, t2):
    """Separating-axis penetration depth of two unit squares (<= 0: apart by -depth)."""
    dx, dy = x2 - x1, y2 - y1
    delta = t2 - t1
    rad = 0.5 + 0.5 * (abs(math.cos(delta)) + abs(math.sin(delta)))
    best = 1e9
    for k in range(4):
        a = t1 if k < 2 else t2
        if k % 2 == 0:
            ax, ay = math.cos(a), math.sin(a)
        else:
            ax, ay = -math.sin(a), math.cos(a)
        o = rad - abs(dx * ax + dy * ay)
        if o < best:
            best = o
    return best


@njit(cache=True)
def candidate_features(px, py, pt, cx, cy, ct, L, frac):
    """Features (C, N_FEAT) and worst overlap (C,) of candidate spots against placed squares."""
    C = cx.shape[0]
    P = px.shape[0]
    f = np.zeros((C, N_FEAT))
    worst = np.zeros(C)
    for c in range(C):
        x, y, t = cx[c], cy[c], ct[c]
        w = 0.5 * (abs(math.cos(t)) + abs(math.sin(t)))
        walls = (x - w, L - x - w, y - w, L - y - w)
        wall_gap = min(min(walls[0], walls[1]), min(walls[2], walls[3]))
        wall_contacts = 0
        for g in walls:
            if g < CONTACT:
                wall_contacts += 1
        gap, contacts, dense, match, deepest = 2.0, 0, 0, 1.0, -1e9
        for p in range(P):
            ddx, ddy = px[p] - x, py[p] - y
            d2 = ddx * ddx + ddy * ddy
            if d2 > 6.25:  # farther than 2.5: cannot touch, outside the density radius
                continue
            if d2 < 2.56:
                dense += 1
            if d2 < 2.0 + 1e-9:
                dep = _pair_depth(x, y, t, px[p], py[p], pt[p])
            else:
                dep = -(math.sqrt(d2) - math.sqrt(2.0))  # lower bound on the gap
            if dep > deepest:
                deepest = dep
            if -dep < gap:
                gap = -dep
                match = math.cos(4.0 * (t - pt[p]))
            if -dep < CONTACT:
                contacts += 1
        worst[c] = max(deepest, -wall_gap)
        f[c, 0] = 2 * x / L - 1
        f[c, 1] = 2 * y / L - 1
        f[c, 2] = math.sqrt(x * x + y * y) / L - 0.7
        f[c, 3] = min(wall_gap, 1.0)
        f[c, 4] = wall_contacts / 2.0
        f[c, 5] = t / (math.pi / 4)
        f[c, 6] = min(gap, 1.0)
        f[c, 7] = contacts / 3.0
        f[c, 8] = dense / 4.0
        f[c, 9] = match
        f[c, 10] = 2 * frac - 1
    return f, worst


class Grid:
    """Candidate spots: centers on a grid of step `step` inside [0, L]^2, times `angles` (degrees)."""

    def __init__(self, L, step=0.1, angles=(0, 10, 20, 30, 40, 45)):
        self.L = L
        cs, xs, ys = [], [], []
        for a in angles:
            t = math.radians(a)
            w = 0.5 * (abs(math.cos(t)) + abs(math.sin(t)))
            g = np.arange(w, L - w + 1e-9, step)
            X, Y = np.meshgrid(g, g, indexing="ij")
            xs.append(X.ravel())
            ys.append(Y.ravel())
            cs.append(np.full(X.size, t))
        self.x, self.y, self.t = np.concatenate(xs), np.concatenate(ys), np.concatenate(cs)


def unpack(genome, hidden):
    p = 0
    w1 = genome[p:p + N_FEAT * hidden].reshape(N_FEAT, hidden); p += N_FEAT * hidden
    b1 = genome[p:p + hidden]; p += hidden
    w2 = genome[p:p + hidden]; p += hidden
    return w1, b1, w2


def genome_size(hidden):
    return N_FEAT * hidden + 2 * hidden


def place_all(genome, n, L, hidden=8, step=0.1, tol=0.02, rng=None):
    """Run the policy: place n squares into [0, L]^2. Returns (x, y, t, leftover overlap)."""
    w1, b1, w2 = unpack(genome, hidden)
    grid = Grid(L, step)
    px, py, pt = np.zeros(0), np.zeros(0), np.zeros(0)
    leftover = 0.0
    for k in range(n):
        f, worst = candidate_features(px, py, pt, grid.x, grid.y, grid.t, L, k / n)
        score = np.tanh(f @ w1 + b1) @ w2
        ok = worst <= tol
        if ok.any():
            score = np.where(ok, score, -np.inf)
            c = int(np.argmax(score))
        else:
            c = int(np.argmin(worst - 1e-3 * score))
            leftover += float(worst[c])
        px, py, pt = np.append(px, grid.x[c]), np.append(py, grid.y[c]), np.append(pt, grid.t[c])
    return px, py, pt, leftover


def develop(genome, n, L, hidden=8, step=0.1, rounds=16, maxiter=400, tol=0.02):
    """Place, compact, squeeze. Returns (fitness, side, (x, y, t))."""
    x, y, t, leftover = place_all(genome, n, L, hidden, step, tol)
    x, y, t, Lc = compact(x, y, t, L, maxiter=maxiter)
    r = squeeze(x, y, t, Lc * 1.0005, rounds=rounds, step=0.001, method="lbfgs", maxiter=1500)
    if r is None:
        worst = float(math.ceil(math.sqrt(n))) * 2
        return worst, worst, None
    return r[3] + 0.01 * leftover, r[3], r[:3]


def _eval(args):
    genome, n, L, hidden, step, tol = args
    return develop(genome, n, L, hidden, step, tol=tol)


def evolve(n, L, gens=40, popsize=None, sigma=0.5, hidden=8, step=0.1, tol=0.02, seed=0, workers=4, log=print):
    """CMA-ES over the placement net for puzzle n with target box L.

    Returns (best_genome, (side, (x, y, t))): the best squeezed packing seen in any genome.
    """
    import cma

    size = genome_size(hidden)
    es = cma.CMAEvolutionStrategy(np.random.default_rng(seed).normal(0, 0.3, size), sigma,
                                  {"popsize": popsize or 4 + int(3 * math.log(size)), "seed": seed + 1,
                                   "verbose": -9})
    best = (math.inf, None)
    best_fit, best_genome = math.inf, None
    with mp.get_context("spawn").Pool(workers) as pool:
        for g in range(gens):
            pop = es.ask()
            res = pool.map(_eval, [(np.asarray(z), n, L, hidden, step, tol) for z in pop])
            fits = [r[0] / math.sqrt(n) for r in res]
            for fit, side, pk in res:
                if pk is not None and side < best[0]:
                    best = (side, pk)
            es.tell(pop, fits)
            if min(fits) < best_fit:
                best_fit, best_genome = min(fits), np.asarray(pop[int(np.argmin(fits))])
            log(f"gen {g:3d}  best {min(fits) * math.sqrt(n):.4f}  median {np.median(fits) * math.sqrt(n):.4f}  "
                f"overall side {best[0]:.4f}  sigma {es.sigma:.3f}", flush=True)
    return best_genome, best
