"""Agents that invent and evolve Game-of-Life packing rules (see `wm.life`).

Each inventor is a rule designer with a theory of what a square should do with its neighbours.
Its theory fixes which local terms it may use (the rest stay at zero) and a hand-written starting
rule; CMA-ES then tunes the free coefficients. A rule's fitness is measured across a set of
training n at once, so only rules that work at every size survive; it is then checked on held-out
n it never saw. Rules, their lineage and their scores go to a JSON library that other agents
(or the swarm's `life` tool) can read and play.

Fitness of rule c on n, from seed packing S with side L_ref:

    r(c, n) = L(c, n) / L_ref

where L(c, n) is the feasible side after the rule plus the standard squeeze finish. The rule's
fitness is the mean of r over the training n (lower is better, < 1 means it beats the seed).
Every feasible packing met during evolution is kept per n, and the best ones can be offered to
the shared store, which accepts only official-metric-checked improvements.
"""

import json
import math
import multiprocessing as mp
import os
import time

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import best_store, life
from .pack_opt import compact, feasible, squeeze

# The schedule terms every inventor tunes; the neighbour terms are what distinguishes them.
SCHEDULE = ["melt", "shrink", "tol", "momentum", "rep"]


@dataclass
class Inventor:
    name: str
    theory: str
    terms: list  # neighbour terms this inventor may use, on top of SCHEDULE
    start: dict = field(default_factory=dict)  # its hand-designed first rule

    @property
    def free(self) -> list:
        return SCHEDULE + [t for t in self.terms if t not in SCHEDULE]


INVENTORS = {
    "granular": Inventor(
        "granular", "Grains under slow compression jam into dense states; off-center contacts make "
        "grains roll, and thermal noise that cools lets a jam unjam into a denser one.",
        ["roll", "noise", "noise_decay"], {"roll": 0.3, "noise": 0.01, "noise_decay": 4.0}),
    "crystal": Inventor(
        "crystal", "Dense packings are locally crystalline: squares should turn to match their "
        "neighbours and slide along shared faces into columns, so domains grow and grain "
        "boundaries carry the slack.", ["align", "slide", "axis"],
        {"align": 0.03, "slide": 0.01, "axis": 0.01}),
    "tension": Inventor(
        "tension", "Surface tension: squares are drawn across small gaps to neighbours and walls, "
        "so slack collects into one hole that the box squeezes out.", ["coh", "coh_range", "wall"],
        {"coh": 0.02, "coh_range": 0.1, "wall": 0.02}),
    "tilt": Inventor(
        "tilt", "The best known packings mix axis-aligned blocks with tilted strips; squares near "
        "crowded walls should turn away from their neighbours (anti-align) and toward 45 deg so "
        "tilted bands can form.", ["align", "axis", "roll"], {"align": -0.02, "axis": -0.005, "roll": 0.2}),
    "spacer": Inventor(
        "spacer", "Holes are the enemy: a soft center-center repulsion spreads squares evenly, so "
        "no region is starved while another is jammed; noise breaks ties.",
        ["space", "space_r", "noise", "noise_decay"], {"space": 0.02, "space_r": 1.3, "noise": 0.005}),
    "open": Inventor(
        "open", "No theory: every term is free and evolution decides.", [n for n in life.NAMES if n not in SCHEDULE]),
}


def rule_from(inv: Inventor, z) -> np.ndarray:
    """Full coefficient vector from the inventor's free genome z (unit-scaled to each range)."""
    c = life.from_dict(inv.start)
    for k, name in enumerate(inv.free):
        i = life.NAMES.index(name)
        c[i] = life.LOW[i] + (life.HIGH[i] - life.LOW[i]) * (0.5 + 0.5 * math.tanh(z[k]))
    return life.decode(c)


def genome_from(inv: Inventor, c) -> np.ndarray:
    z = []
    for name in inv.free:
        i = life.NAMES.index(name)
        u = (c[i] - life.LOW[i]) / (life.HIGH[i] - life.LOW[i])
        z.append(math.atanh(np.clip(2 * u - 1, -0.999, 0.999)))
    return np.array(z)


# ----------------------------------------------------------------------------- playing a rule on n


def seed_packing(n: int, mode: str, rng, root=None):
    """(x, y, t, L) to start from: the stored best ('store'), the stored n+1 packing with one random
    square taken out ('drop': the rule has a hole to close), or a loose random layout ('random')."""
    from .swarm.tools import from_xyd
    if mode == "store":
        xyd = best_store.load_xyd(n, root)
        if xyd is not None:
            p = from_xyd(*xyd)
            return p.x, p.y, p.t, p.L
    if mode == "drop":
        xyd = best_store.load_xyd(n + 1, root)
        if xyd is not None:
            p = from_xyd(*xyd)
            keep = np.delete(np.arange(n + 1), rng.integers(n + 1))
            return p.x[keep], p.y[keep], p.t[keep], p.L
        return seed_packing(n, "store", rng, root)
    return life.random_seed(n, rng)


def finish(x, y, t, L):
    """Penalty compact (side free) then exact squeeze: a feasible packing, or None."""
    x, y, t, L2 = compact(x, y, t, L, mus=(100.0, 1e3, 1e4, 1e5), maxiter=300)
    r = squeeze(x, y, t, L2 * 1.0005, rounds=25, step=0.001)
    if r is None or not feasible(*r):
        return None
    return r


def play_on(c, n, mode="store", steps=600, seed=0, root=None):
    """Play rule c on n. Returns (side, (x, y, t) or None, reference side)."""
    rng = np.random.default_rng(seed)
    x, y, t, L = seed_packing(n, mode, rng, root)
    ref = best_store.best_side(n, root)
    rx, ry, rt, rL = life.run_rule(c, x, y, t, L, steps=steps, seed=seed)
    r = finish(rx, ry, rt, rL)
    if r is None:
        return math.inf, None, ref
    return float(r[3]), (r[0], r[1], r[2]), ref


def _job(args):
    c, n, mode, steps, seed, root = args
    side, pk, ref = play_on(c, n, mode, steps, seed, root)
    return n, side, pk, ref


# ----------------------------------------------------------------------------- evolution


def evolve(inv: Inventor, train, mode="store", gens=20, popsize=10, sigma=0.6, steps=600, seed=0,
           workers=4, root=None, log=print, tries=2):
    """CMA-ES over the inventor's free terms; fitness = mean over `train` of the best side ratio
    of `tries` runs (so a rule that sometimes finds a new basin beats one that never moves).

    Returns (best rule, best fitness, per-n best packings {n: (side, x, y, t)}).
    """
    import cma

    z0 = genome_from(inv, life.from_dict(inv.start))
    es = cma.CMAEvolutionStrategy(z0, sigma, {"popsize": popsize, "seed": seed + 1, "verbose": -9})
    best = (math.inf, None)
    per_n = {}
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as pool:
        for g in range(gens):
            zs = es.ask()
            rules = [rule_from(inv, z) for z in zs]
            jobs = [(c, n, mode, steps, seed * 100000 + g * 100 + s, root)
                    for c in rules for n in train for s in range(tries)]
            out = pool.map(_job, jobs)
            fits = []
            m = len(train) * tries
            for k, c in enumerate(rules):
                rs = {}
                for n, side, pk, ref in out[k * m:(k + 1) * m]:
                    rs[n] = min(rs.get(n, 1.2), side / ref)
                    if pk is not None and side < per_n.get(n, (math.inf,))[0]:
                        per_n[n] = (side, *pk)
                fits.append(float(np.mean(list(rs.values()))))
            es.tell(zs, fits)
            k = int(np.argmin(fits))
            if fits[k] < best[0]:
                best = (fits[k], rules[k])
            log(f"[{inv.name}] gen {g:2d}  best {min(fits):.5f}  mean {np.mean(fits):.5f}  overall {best[0]:.5f}")
    return best[1], best[0], per_n


def score_rule(c, ns, mode="store", steps=600, seeds=(0,), workers=4, root=None):
    """Play a fixed rule on every n (best over seeds). Returns {n: (side, ref, packing)}."""
    jobs = [(c, n, mode, steps, s, root) for n in ns for s in seeds]
    ctx = mp.get_context("spawn")
    res = {}
    with ctx.Pool(workers) as pool:
        for n, side, pk, ref in pool.imap_unordered(_job, jobs):
            if n not in res or side < res[n][0]:
                res[n] = (side, ref, pk)
    return res


def offer_all(per_n, source, root=None):
    """Offer each per-n packing that beats the store. Returns the list of accepted results."""
    won = []
    for n, (side, *pk) in sorted(per_n.items()):
        if pk[0] is None or side >= best_store.best_side(n, root) - 1e-9:
            continue
        x, y, t = pk
        r = best_store.offer(n, x, y, np.rad2deg(t), source, root=root)
        if r["accepted"]:
            won.append((n, r["previous"], r["side"]))
    return won


# ----------------------------------------------------------------------------- rule library

LIBRARY = Path(os.environ.get("LIFE_RULES", Path(__file__).resolve().parent.parent / "rules" / "life_rules.json"))


def library_rule(inventor: str, path=None) -> np.ndarray:
    """The best rule an inventor has in the library (by held-out score), else its hand-written start."""
    path = Path(path or LIBRARY)
    lib = json.loads(path.read_text()) if path.exists() else []
    mine = [r for r in lib if r["inventor"] == inventor]
    if not mine:
        return life.decode(life.from_dict(INVENTORS[inventor].start))
    best = min(mine, key=lambda r: r["held_out"]["mean"] if r.get("held_out") else r["fitness"])
    return life.from_dict(best["rule"])


def save_rule(path, name, inv: Inventor, c, fitness, train, mode, held=None, parent=None):
    """Append a rule with its lineage and scores to the JSON library at path."""
    lib = json.loads(open(path).read()) if os.path.exists(path) else []
    lib.append({"name": name, "inventor": inv.name, "theory": inv.theory, "parent": parent,
                "mode": mode, "train": list(train), "fitness": fitness, "held_out": held,
                "rule": life.as_dict(c), "made": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(lib, f, indent=1)
    os.replace(tmp, path)
    return lib
