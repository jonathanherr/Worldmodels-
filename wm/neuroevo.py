"""Neuroevolution solver for Squares in Squares.

A small MLP is a developmental encoding of a packing: for every square k of puzzle n it reads a
feature vector (where k would sit in the ceil(sqrt(n)) grid, its rank k/n, a few Fourier terms,
two fixed per-square noise values to break symmetry, and features of n itself) and outputs an
offset from its grid cell and a rotation. With zero output weights the net decodes to the grid.

CMA-ES evolves the net's weights. A genome's fitness is Baldwinian: decode the poses, run the
local optimizer in `wm.pack_opt` (`compact_squeeze`) for a short budget, and score the smallest feasible
box it reaches. The local optimizer handles the fine geometry and the network only has to put
the squares in a good basin. One net can be shared across several n, so patterns that work for
one size can transfer to its neighbours.

The best packing seen for each n (from any genome) is refined by basin hopping (`refine`) and
offered to the shared best-per-n store, which accepts it only if Kaggle's metric certifies it
and it beats what is stored.
"""

import math
import multiprocessing as mp
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
from dataclasses import dataclass

import numpy as np

from .pack_opt import compact_squeeze

N_FOURIER = 3
N_NOISE = 2


def features(n: int, base=None) -> np.ndarray:
    """(n, F) input features for the squares of puzzle n.

    With a base packing (x, y, t, L) the first two features are its normalized centers instead of
    grid cells, and the first noise feature becomes its angle.
    """
    m = math.ceil(math.sqrt(n))
    k = np.arange(n)
    col, row = (k % m + 0.5) / m, (k // m + 0.5) / m
    if base is not None:
        col, row = base[0] / base[3], base[1] / base[3]
    u = (k + 0.5) / n
    four = [f(2 * np.pi * (i + 1) * u) for i in range(N_FOURIER) for f in (np.sin, np.cos)]
    rng = np.random.default_rng(1_000_003 * n + 17)  # fixed per n, so a genome always decodes the same
    noise = rng.uniform(-1, 1, (N_NOISE, n))
    if base is not None:
        noise[0] = base[2] / (np.pi / 4)
    nfeat = [np.full(n, math.log(n) / math.log(1000)), np.full(n, m - math.sqrt(n)),
             np.full(n, (n - (m - 1) ** 2) / (2 * m - 1))]  # how full the last grid ring is
    cols = [2 * col - 1, 2 * row - 1, 2 * u - 1, *four, *noise, *nfeat]
    return np.stack(cols, -1)


N_IN = 3 + 2 * N_FOURIER + N_NOISE + 3


@dataclass
class NetShape:
    hidden: int = 16
    n_in: int = N_IN
    n_out: int = 3

    @property
    def size(self) -> int:
        # Two layers plus one global gene for the decode scale.
        return self.n_in * self.hidden + self.hidden + self.hidden * self.n_out + self.n_out + 1


def decode(genome: np.ndarray, n: int, shape: NetShape, base=None):
    """Genome -> (x, y, t radians, L0): poses in a box of side L0, ready for `compact_squeeze`.

    Without a base the net deforms the grid. With a base packing (x, y, t, L), e.g. the stored
    best, it deforms that packing instead (offsets up to half a square, turns up to 22.5 deg), so
    evolution searches the neighbourhood of the best known answer.
    """
    h, i, o = shape.hidden, shape.n_in, shape.n_out
    p = 0
    w1 = genome[p:p + i * h].reshape(i, h); p += i * h
    b1 = genome[p:p + h]; p += h
    w2 = genome[p:p + h * o].reshape(h, o); p += h * o
    b2 = genome[p:p + o]; p += o
    scale_gene = genome[p]
    m = math.ceil(math.sqrt(n))
    feats = features(n, base)
    out = np.tanh(np.tanh(feats @ w1 + b1) @ w2 + b2)
    if base is not None:
        bx, by, bt, bL = base
        L0 = bL * math.exp(0.05 * math.tanh(scale_gene))
        k = L0 / bL
        return bx * k + 0.5 * out[:, 0], by * k + 0.5 * out[:, 1], bt + out[:, 2] * (np.pi / 8), L0
    # Box between sqrt(n) and the grid side + a little; offsets up to one cell; angles in +-45 deg.
    L0 = math.sqrt(n) + (m + 0.3 - math.sqrt(n)) / (1 + math.exp(-scale_gene))
    cell = L0 / m
    x = (feats[:, 0] + 1) / 2 * L0 + out[:, 0] * cell
    y = (feats[:, 1] + 1) / 2 * L0 + out[:, 1] * cell
    t = out[:, 2] * (np.pi / 4)
    return x, y, t, L0


def develop(genome, n, shape, rounds=16, maxiter=400, base=None):
    """Decode and locally optimize. Returns (side, (x, y, t)) of the squeezed packing."""
    x, y, t, L0 = decode(genome, n, shape, base)
    r = compact_squeeze(x, y, t, L0, rounds=rounds, maxiter=maxiter)
    if r is None:
        return float(math.ceil(math.sqrt(n))) * 2, None
    x, y, t, L = r
    return L, (x, y, t)


def _eval(args):
    genome, ns, shape, rounds, maxiter, bases = args
    return [develop(genome, n, shape, rounds, maxiter, bases.get(n)) for n in ns]


def refine(x, y, t, L, iters=60, seed=0, frac=0.15, maxiter=400):
    """Basin hopping: jolt a few squares, re-squeeze from the current box, keep improvements."""
    rng = np.random.default_rng(seed)
    best = (np.array(x), np.array(y), np.array(t), L)
    n = len(x)
    for _ in range(iters):
        bx, by, bt, bL = best
        nx, ny, nt = bx.copy(), by.copy(), bt.copy()
        k = max(1, int(rng.binomial(n, frac)))
        idx = rng.choice(n, k, replace=False)
        if rng.random() < 0.5:  # small shake of a few squares
            nx[idx] += rng.normal(0, 0.3, k)
            ny[idx] += rng.normal(0, 0.3, k)
            nt[idx] += rng.normal(0, 0.3, k)
        else:  # move one square to a random spot and give it a random tilt
            j = idx[0]
            nx[j], ny[j], nt[j] = rng.uniform(0, bL), rng.uniform(0, bL), rng.uniform(-np.pi / 4, np.pi / 4)
        r = compact_squeeze(nx, ny, nt, bL, rounds=30, maxiter=maxiter)
        if r is not None and r[3] < bL - 1e-9:
            best = r
    return best


def random_genome_baseline(n, evals, seed=0, rounds=16, maxiter=400):
    """Control: squeeze uniformly random layouts, same per-evaluation budget as a genome."""
    rng = np.random.default_rng(seed)
    best = (math.inf, None)
    for _ in range(evals):
        L0 = math.sqrt(n) * rng.uniform(1.05, 1.4)
        r = compact_squeeze(rng.uniform(0, L0, n), rng.uniform(0, L0, n), rng.uniform(-np.pi / 4, np.pi / 4, n),
                            L0, rounds=rounds, maxiter=maxiter)
        if r is not None and r[3] < best[0]:
            best = (r[3], r[:3])
    return best


def evolve(ns, gens=60, popsize=None, sigma=0.5, hidden=16, seed=0, workers=4, rounds=16, maxiter=400,
           bases=None, log=print):
    """CMA-ES over one shared net for the sizes in ns.

    Fitness is the mean of side / sqrt(n) over ns. Returns (best_genome, best_per_n) where
    best_per_n maps n -> (side, (x, y, t)), the best squeezed packing seen for n in any genome.
    bases maps n -> (x, y, t, L), a packing for the net to deform instead of the grid.
    """
    bases = bases or {}
    import cma

    shape = NetShape(hidden=hidden)
    es = cma.CMAEvolutionStrategy(np.zeros(shape.size), sigma,
                                  {"popsize": popsize or 4 + int(3 * math.log(shape.size)), "seed": seed + 1,
                                   "verbose": -9})
    best_per_n = {n: (math.inf, None) for n in ns}
    best_fit, best_genome = math.inf, None
    # Spawned workers with one BLAS thread each: forked children of a process that already
    # started BLAS threads can spin forever.
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as pool:
        for g in range(gens):
            pop = es.ask()
            results = pool.map(_eval, [(np.asarray(z), ns, shape, rounds, maxiter, bases) for z in pop])
            fits = []
            for z, res in zip(pop, results):
                fits.append(float(np.mean([side / math.sqrt(n) for n, (side, _) in zip(ns, res)])))
                for n, (side, pk) in zip(ns, res):
                    if pk is not None and side < best_per_n[n][0]:
                        best_per_n[n] = (side, pk)
            es.tell(pop, fits)
            if min(fits) < best_fit:
                best_fit, best_genome = min(fits), np.asarray(pop[int(np.argmin(fits))])
            log(f"gen {g:3d}  best {min(fits):.5f}  mean {np.mean(fits):.5f}  overall {best_fit:.5f}  "
                f"sigma {es.sigma:.3f}  " + " ".join(f"{n}:{best_per_n[n][0]:.4f}" for n in ns))
    return best_genome, best_per_n, shape
