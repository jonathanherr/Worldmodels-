"""NEAT-style neuroevolution for Squares in Squares: topology, activations and weights all evolve.

Where `wm.neuroevo` evolves only the weights of a fixed MLP with CMA-ES, this evolves the network
itself, in the manner of NEAT (Stanley & Miikkulainen 2002) with CPPN-style activation choice:

* A genome is a set of node genes (hidden and output nodes, each with an activation function and
  a bias) and connection genes (source, target, weight, enabled, innovation number), plus one
  box-size gene. Inputs are the per-square features from `wm.neuroevo.features`.
* Mutation operators: perturb or reset weights and biases, add a connection, split a connection
  with a new node, swap a hidden node's activation function, toggle a connection on or off, and
  nudge the box gene.
* Crossover aligns connection genes by innovation number: matching genes come from either
  parent at random, disjoint and excess genes from the fitter one.
* Speciation by compatibility distance (disjoint/excess genes and mean weight difference), with
  a threshold that adapts to keep about `target_species` species. Offspring are shared between
  species by their mean fitness rank, so new structure gets time to tune its weights before it
  has to compete with the whole population.

The net runs once per square (vectorized over squares), and its three outputs go through the
same `place` mapping and Baldwinian fitness as the CMA-ES solver: decode, `compact_squeeze`,
score the box. A genome with no connections decodes to the grid (or to the stored packing with
`base`), which is where evolution starts.
"""

import math
import multiprocessing as mp
import os
from dataclasses import dataclass, field

import numpy as np

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

from .neuroevo import N_IN, features, place  # noqa: E402
from .pack_opt import compact, squeeze  # noqa: E402

N_OUT = 3
OUTPUTS = tuple(range(N_IN, N_IN + N_OUT))  # node ids: inputs are 0..N_IN-1, then the outputs

ACTIVATIONS = {
    "tanh": np.tanh,
    "sin": lambda v: np.sin(np.pi * v),
    "gauss": lambda v: np.exp(-v * v),
    "abs": lambda v: np.abs(v) - 0.5,
    "relu": lambda v: np.maximum(v, 0.0),
    "step": lambda v: np.tanh(4.0 * v),
    "identity": lambda v: v,
}
ACT_NAMES = tuple(ACTIVATIONS)


@dataclass
class Conn:
    src: int
    dst: int
    w: float
    on: bool = True


@dataclass
class Genome:
    nodes: dict = field(default_factory=dict)  # node id -> [activation name, bias]
    conns: dict = field(default_factory=dict)  # innovation number -> Conn
    scale: float = 0.0  # box-size gene, as in wm.neuroevo
    fitness: float = math.inf  # mean side / sqrt(n), lower is better

    def copy(self):
        return Genome({k: list(v) for k, v in self.nodes.items()},
                      {k: Conn(c.src, c.dst, c.w, c.on) for k, c in self.conns.items()}, self.scale)

    def complexity(self):
        return len(self.nodes) - N_OUT, sum(c.on for c in self.conns.values())


class Innovations:
    """Hands out the same innovation number to the same structural change within a run."""

    def __init__(self):
        self.conn = {}  # (src, dst) -> innovation
        self.split = {}  # innovation of the split connection -> new node id
        self.next_node = N_IN + N_OUT

    def conn_id(self, src, dst):
        return self.conn.setdefault((src, dst), len(self.conn))

    def node_for_split(self, innov):
        if innov not in self.split:
            self.split[innov] = self.next_node
            self.next_node += 1
        return self.split[innov]


# ----------------------------------------------------------------------------- evaluation


def _order(g: Genome):
    """Hidden and output nodes in an order where every node comes after its enabled inputs."""
    incoming = {k: [] for k in g.nodes}
    for c in g.conns.values():
        if c.on:
            incoming[c.dst].append(c)
    order, done = [], set(range(N_IN))
    pending = set(g.nodes)
    while pending:
        ready = [k for k in pending if all(c.src in done for c in incoming[k])]
        if not ready:  # cannot happen for genomes built by the operators below
            raise ValueError("cycle in genome")
        for k in sorted(ready):
            order.append(k)
            done.add(k)
            pending.discard(k)
    return order, incoming


def activate(g: Genome, feats: np.ndarray) -> np.ndarray:
    """Run the network on every square at once: (n, N_IN) features -> (n, 3) outputs in [-1, 1]."""
    order, incoming = _order(g)
    vals = {i: feats[:, i] for i in range(N_IN)}
    for k in order:
        act, bias = g.nodes[k]
        s = np.full(len(feats), bias)
        for c in incoming[k]:
            s = s + c.w * vals[c.src]
        vals[k] = ACTIVATIONS[act](s)
    return np.stack([np.tanh(vals[k]) if k in OUTPUTS else vals[k] for k in OUTPUTS], -1)


def decode(g: Genome, n: int, base=None):
    feats = features(n, base)
    return place(activate(g, feats), feats, n, g.scale, base)


def develop(g, n, rounds=16, maxiter=400, base=None, smooth=0.0):
    """Decode, compact and squeeze. Returns (fitness side, side, (x, y, t)).

    The fitness side is the feasible side, optionally blended with the side the penalty
    compaction reached (slightly infeasible, but continuous): many layouts squeeze back to exactly
    the grid side, and the blend tells apart the ones that nearly got tighter.
    """
    x, y, t, L0 = decode(g, n, base)
    x, y, t, Lc = compact(x, y, t, L0, maxiter=maxiter)
    r = squeeze(x, y, t, Lc * 1.0005, rounds=rounds, step=0.001, method="lbfgs", maxiter=1500)
    if r is None:
        worst = float(math.ceil(math.sqrt(n))) * 2
        return worst, worst, None
    return (1 - smooth) * r[3] + smooth * min(Lc, r[3]), r[3], r[:3]


def _eval(args):
    g, ns, rounds, maxiter, bases, smooth = args
    return [develop(g, n, rounds, maxiter, bases.get(n), smooth) for n in ns]


# ----------------------------------------------------------------------------- operators


@dataclass
class NeatConfig:
    pop: int = 48
    target_species: int = 6
    p_weight: float = 0.8  # chance a child's weights are perturbed
    weight_sigma: float = 0.4
    p_reset: float = 0.1  # per weight, when perturbing: draw a fresh value instead
    p_add_conn: float = 0.25
    p_add_node: float = 0.12
    p_act: float = 0.1
    p_toggle: float = 0.05
    p_crossover: float = 0.6
    init_conn_prob: float = 0.15  # each input -> output connection present in the first population
    init_weight: float = 0.5
    elite_per_species: int = 1
    survive_frac: float = 0.5  # parents come from the better half of each species
    c_disjoint: float = 1.0
    c_weight: float = 0.5
    stagnation: int = 15  # a species that has not improved for this many generations stops breeding


def initial_genome(rng, inn: Innovations, cfg: NeatConfig) -> Genome:
    g = Genome({k: ["identity", 0.0] for k in OUTPUTS})
    for src in range(N_IN):
        for dst in OUTPUTS:
            if rng.random() < cfg.init_conn_prob:
                g.conns[inn.conn_id(src, dst)] = Conn(src, dst, rng.normal(0, cfg.init_weight))
    g.scale = rng.normal(0, 0.5)
    return g


def _creates_cycle(g: Genome, src, dst):
    """Would src -> dst close a loop? True if dst already reaches src."""
    out = {}
    for c in g.conns.values():
        out.setdefault(c.src, []).append(c.dst)
    stack, seen = [dst], set()
    while stack:
        k = stack.pop()
        if k == src:
            return True
        if k not in seen:
            seen.add(k)
            stack.extend(out.get(k, []))
    return False


def mutate(g: Genome, rng, inn: Innovations, cfg: NeatConfig) -> Genome:
    g = g.copy()
    if rng.random() < cfg.p_weight:
        for c in g.conns.values():
            c.w = rng.normal(0, 1.0) if rng.random() < cfg.p_reset else c.w + rng.normal(0, cfg.weight_sigma)
        for node in g.nodes.values():
            node[1] += rng.normal(0, cfg.weight_sigma / 2)
        g.scale += rng.normal(0, 0.2)
    if rng.random() < cfg.p_add_conn:
        hidden = [k for k in g.nodes if k not in OUTPUTS]
        for _ in range(20):
            src = int(rng.choice(list(range(N_IN)) + hidden))
            dst = int(rng.choice(hidden + list(OUTPUTS)))
            if src == dst or _creates_cycle(g, src, dst):
                continue
            innov = inn.conn_id(src, dst)
            if innov in g.conns:
                g.conns[innov].on = True
                continue
            g.conns[innov] = Conn(src, dst, rng.normal(0, cfg.init_weight))
            break
    if rng.random() < cfg.p_add_node and g.conns:
        on = [k for k, c in g.conns.items() if c.on]
        if on:
            innov = int(rng.choice(on))
            c = g.conns[innov]
            new = inn.node_for_split(innov)
            if new not in g.nodes:
                c.on = False
                g.nodes[new] = [str(rng.choice(ACT_NAMES)), 0.0]
                g.conns[inn.conn_id(c.src, new)] = Conn(c.src, new, 1.0)
                g.conns[inn.conn_id(new, c.dst)] = Conn(new, c.dst, c.w)
    if rng.random() < cfg.p_act:
        hidden = [k for k in g.nodes if k not in OUTPUTS]
        if hidden:
            g.nodes[int(rng.choice(hidden))][0] = str(rng.choice(ACT_NAMES))
    if rng.random() < cfg.p_toggle and g.conns:
        c = g.conns[int(rng.choice(list(g.conns)))]
        if c.on or not _creates_cycle(g, c.src, c.dst):
            c.on = not c.on
    return g


def crossover(a: Genome, b: Genome, rng) -> Genome:
    """a is the fitter parent. Matching genes from either at random, the rest from a."""
    child = Genome(scale=a.scale if rng.random() < 0.5 else b.scale)
    for innov, ca in a.conns.items():
        cb = b.conns.get(innov)
        c = cb if (cb is not None and rng.random() < 0.5) else ca
        on = ca.on if cb is None else (ca.on and cb.on) or rng.random() < 0.25
        child.conns[innov] = Conn(c.src, c.dst, c.w, on)
    for k, node in a.nodes.items():
        other = b.nodes.get(k)
        child.nodes[k] = list(other if (other is not None and rng.random() < 0.5) else node)
    # Enabled flags mixed from two parents can close a loop; fall back to the fitter parent's.
    try:
        _order(child)
    except ValueError:
        for innov, c in child.conns.items():
            c.on = a.conns[innov].on
    return child


def distance(a: Genome, b: Genome, cfg: NeatConfig) -> float:
    ka, kb = set(a.conns), set(b.conns)
    shared = ka & kb
    n = max(len(ka), len(kb), 1)
    disjoint = len(ka ^ kb)
    wdiff = np.mean([abs(a.conns[k].w - b.conns[k].w) for k in shared]) if shared else 0.0
    acts = sum(a.nodes[k][0] != b.nodes[k][0] for k in set(a.nodes) & set(b.nodes))
    return cfg.c_disjoint * (disjoint + acts) / n + cfg.c_weight * wdiff


# ----------------------------------------------------------------------------- evolution


@dataclass
class Species:
    rep: Genome
    members: list = field(default_factory=list)
    best: float = math.inf
    since_improved: int = 0


def evolve(ns, gens=40, cfg: NeatConfig | None = None, seed=0, workers=4, rounds=16, maxiter=400,
           bases=None, smooth=0.0, log=print):
    """NEAT over one population for the sizes in ns (fitness: mean side / sqrt(n), lower is better).

    Returns (best_genome, best_per_n) where best_per_n maps n -> (side, (x, y, t)), the best
    squeezed packing seen for n in any genome.
    """
    cfg = cfg or NeatConfig()
    bases = bases or {}
    rng = np.random.default_rng(seed)
    inn = Innovations()
    pop = [initial_genome(rng, inn, cfg) for _ in range(cfg.pop)]
    pop[0] = Genome({k: ["identity", 0.0] for k in OUTPUTS})  # the plain grid / stored packing
    species: list[Species] = []
    threshold = 1.0
    best_per_n = {n: (math.inf, None) for n in ns}
    champion = None
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as pool:
        for gen in range(gens):
            results = pool.map(_eval, [(g, ns, rounds, maxiter, bases, smooth) for g in pop])
            for g, res in zip(pop, results):
                g.fitness = float(np.mean([fit / math.sqrt(n) for n, (fit, _, _) in zip(ns, res)]))
                for n, (_, side, pk) in zip(ns, res):
                    if pk is not None and side < best_per_n[n][0]:
                        best_per_n[n] = (side, pk)
            gen_best = min(pop, key=lambda g: g.fitness)
            if champion is None or gen_best.fitness < champion.fitness:
                champion = gen_best.copy()
                champion.fitness = gen_best.fitness

            # Speciate against last generation's representatives.
            for s in species:
                s.members = []
            for g in pop:
                for s in species:
                    if distance(g, s.rep, cfg) < threshold:
                        s.members.append(g)
                        break
                else:
                    species.append(Species(rep=g, members=[g]))
            species = [s for s in species if s.members]
            threshold *= 1.1 if len(species) > cfg.target_species else 0.9 if len(species) < cfg.target_species else 1.0
            threshold = max(threshold, 0.1)
            for s in species:
                s.members.sort(key=lambda g: g.fitness)
                if s.members[0].fitness < s.best - 1e-9:
                    s.best, s.since_improved = s.members[0].fitness, 0
                else:
                    s.since_improved += 1
                s.rep = s.members[int(rng.integers(len(s.members)))]
            breeding = [s for s in species if s.since_improved < cfg.stagnation] or \
                [min(species, key=lambda s: s.best)]

            # Offspring per species from the mean fitness rank of its members (rank 1 = best).
            ranked = sorted(pop, key=lambda g: g.fitness)
            score = {id(g): (len(pop) - r) / len(pop) for r, g in enumerate(ranked)}
            weights = np.array([np.mean([score[id(g)] for g in s.members]) for s in breeding])
            quota = np.floor(weights / weights.sum() * cfg.pop).astype(int)
            quota[np.argmax(weights)] += cfg.pop - quota.sum()

            nxt = [champion.copy()]
            for s, q in zip(breeding, quota):
                kept = s.members[:min(cfg.elite_per_species, q)]
                nxt.extend(g.copy() for g in kept)
                parents = s.members[:max(1, int(math.ceil(len(s.members) * cfg.survive_frac)))]
                for _ in range(q - len(kept)):
                    a = parents[int(rng.integers(len(parents)))]
                    if len(parents) > 1 and rng.random() < cfg.p_crossover:
                        b = parents[int(rng.integers(len(parents)))]
                        a, b = (a, b) if a.fitness <= b.fitness else (b, a)
                        child = crossover(a, b, rng)
                    else:
                        child = a
                    nxt.append(mutate(child, rng, inn, cfg))
            pop = nxt[:cfg.pop]

            hid, con = champion.complexity()
            log(f"gen {gen:3d}  best {gen_best.fitness:.5f}  champion {champion.fitness:.5f}  "
                f"species {len(species)}  champion nodes {hid} conns {con}  "
                + " ".join(f"{n}:{best_per_n[n][0]:.4f}" for n in ns), flush=True)
    return champion, best_per_n


def describe(g: Genome) -> str:
    """Human-readable summary of a genome's structure."""
    names = ["col", "row", "rank"] + [f"{f}{i + 1}" for i in range(3) for f in ("sin", "cos")] + \
        ["noise0", "noise1", "logn", "gap", "ring"]
    out_names = {OUTPUTS[0]: "dx", OUTPUTS[1]: "dy", OUTPUTS[2]: "tilt"}

    def name(k):
        return names[k] if k < N_IN else out_names.get(k, f"h{k}({g.nodes[k][0]})")

    lines = [f"{name(c.src)} -> {name(c.dst)}  w={c.w:+.2f}" for c in g.conns.values() if c.on]
    return "\n".join(lines) + f"\nbox gene {g.scale:+.2f}"
