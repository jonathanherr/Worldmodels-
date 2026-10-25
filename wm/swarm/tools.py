"""Composable packing tools for the research swarm, all behind one interface.

A packing in flight is a `Packing`: centers x, y, angles t (radians) and a box side L, with the
squares meant to sit in [0, L]^2. It may overlap while a recipe is running; only the final
packing has to be feasible.

Every tool is a function `tool(ctx, state, **args) -> Packing`, registered with a kind:

* construct  ignores `state` and builds a packing for ctx.n (grid, store, random, tilted block,
             a bigger stored packing with squares dropped, a smaller one wrapped in an L, ...)
* perturb    moves some squares (shake, relocate to holes, rotate field, shrink the box)
* relax      local optimization from `wm.pack_opt` / `wm.lift` (relax, compact, squeeze, lift)
* combine    merges `state` with a second packing built by a sub-recipe (crossover)
* meta       runs a sub-recipe several times (repeat with Metropolis acceptance, best_of)

A recipe is a JSON list of steps, `[{"t": "store"}, {"t": "jolt", "frac": 0.1}, {"t": "squeeze"}]`,
and meta/combine steps carry nested recipes under "body" / "other". `run_recipe` executes one for
a given n, makes the result feasible, and `evaluate` offers it to the shared best-per-n store,
which keeps it only if Kaggle's metric certifies it and it beats what is stored. Recipes are plain
data so agents can post them on the board, copy each other's, and splice them together.
"""

import hashlib
import json
import math
import time
from dataclasses import dataclass, field

import numpy as np

from .. import best_store
from ..constructions import candidate_sides, configs, l_wrap, tilted_block
from ..pack_opt import MARGIN, compact, energy_grad, feasible, relax_fire, squeeze
from ..squares import corners, puzzle_score


@dataclass
class Packing:
    x: np.ndarray
    y: np.ndarray
    t: np.ndarray  # radians
    L: float

    @property
    def n(self) -> int:
        return len(self.x)

    def copy(self) -> "Packing":
        return Packing(self.x.copy(), self.y.copy(), self.t.copy(), float(self.L))


@dataclass
class Ctx:
    """What a tool may use: the puzzle size, randomness, a wall-clock deadline, the store."""
    n: int
    rng: np.random.Generator
    deadline: float = math.inf
    store_root: object = None  # None = $SQUARES_BEST_DIR / repo solutions
    log: list = field(default_factory=list)

    def time_left(self) -> float:
        return self.deadline - time.time()


TOOLS: dict = {}


def tool(kind: str, doc: str, **params):
    """Register a tool. params maps argument name -> (default, low, high) for numeric ranges or
    (default, choices) for categorical ones; agents use the ranges to mutate recipes."""
    def wrap(fn):
        TOOLS[fn.__name__] = {"fn": fn, "kind": kind, "doc": doc, "params": params}
        return fn
    return wrap


def from_xyd(x, y, deg, L=None) -> Packing:
    """Packing from centers and degrees, moved so its bounding box starts at the origin."""
    x, y, deg = (np.asarray(a, float) for a in (x, y, deg))
    c = corners(x, y, deg).reshape(-1, 2)
    lo, hi = c.min(0), c.max(0)
    side = float((hi - lo).max())
    return Packing(x - lo[0], y - lo[1], np.deg2rad(deg), max(side, L or 0.0))


def is_feasible(p: Packing, margin: float = 0.0) -> bool:
    """No overlap and inside the box. Margin 0 accepts stored packings (whose gaps are ~1e-9);
    the store's official check has the last word."""
    return bool(feasible(p.x, p.y, p.t, p.L, margin))


def _drop(p: Packing, k: int, how: str, rng) -> Packing:
    """Remove k squares: random ones, or the ones nearest the top/right walls (a box can then
    shrink from that side), or the most crowded ones (largest overlap pressure)."""
    if k <= 0:
        return p
    if how == "wall":
        key = np.maximum(p.x, p.y) + rng.uniform(0, 1e-3, p.n)
        idx = np.argsort(-key)[:k]
    elif how == "crowded":
        d2 = (p.x[:, None] - p.x[None]) ** 2 + (p.y[:, None] - p.y[None]) ** 2
        crowd = (d2 < 1.6).sum(1) + rng.uniform(0, 0.5, p.n)
        idx = np.argsort(-crowd)[:k]
    else:
        idx = rng.choice(p.n, k, replace=False)
    keep = np.setdiff1d(np.arange(p.n), idx)
    return Packing(p.x[keep], p.y[keep], p.t[keep], p.L)


def _holes(p: Packing, k: int, rng, res: int = 0):
    """k points of the box farthest from every center (the biggest holes), greedily."""
    res = res or max(16, int(4 * p.L))
    g = (np.arange(res) + 0.5) / res * p.L
    gx, gy = np.meshgrid(g, g)
    gx, gy = gx.ravel(), gy.ravel()
    w = 0.5
    ok = (gx >= w) & (gx <= p.L - w) & (gy >= w) & (gy <= p.L - w)
    gx, gy = gx[ok], gy[ok]
    if len(gx) == 0:
        return rng.uniform(0, p.L, (k, 2))
    cx, cy = list(p.x), list(p.y)
    out = []
    dmin = np.full(len(gx), np.inf)
    for a, b in zip(cx, cy):
        dmin = np.minimum(dmin, (gx - a) ** 2 + (gy - b) ** 2)
    for _ in range(k):
        j = int(np.argmax(dmin + rng.uniform(0, 1e-6, len(dmin))))
        out.append((gx[j], gy[j]))
        dmin = np.minimum(dmin, (gx - gx[j]) ** 2 + (gy - gy[j]) ** 2)
    return np.array(out)


def _add(p: Packing, k: int, rng) -> Packing:
    """Add k squares at the biggest holes with random tilt."""
    if k <= 0:
        return p
    h = _holes(p, k, rng)
    return Packing(np.concatenate([p.x, h[:, 0]]), np.concatenate([p.y, h[:, 1]]),
                   np.concatenate([p.t, rng.uniform(-np.pi / 4, np.pi / 4, k)]), p.L)


def fit_count(p: Packing, n: int, rng, how: str = "wall") -> Packing:
    return _drop(p, p.n - n, how, rng) if p.n > n else _add(p, n - p.n, rng)


# ----------------------------------------------------------------------------- construct


@tool("construct", "Axis-aligned ceil(sqrt n) grid.")
def grid(ctx, state):
    k = math.ceil(math.sqrt(ctx.n))
    i = np.arange(ctx.n)
    return Packing(i % k + 0.5, i // k + 0.5, np.zeros(ctx.n), float(k))


@tool("construct", "The best certified packing in the shared store (grid if none).")
def store(ctx, state):
    xyd = best_store.load_xyd(ctx.n, ctx.store_root)
    return grid(ctx, None) if xyd is None else from_xyd(*xyd)


@tool("construct", "Uniformly random centers and tilts in a loose box of side sqrt(n) * loose.",
      loose=(1.25, 1.05, 1.6))
def random(ctx, state, loose=1.25):
    L = math.sqrt(ctx.n) * loose
    r = ctx.rng
    return Packing(r.uniform(0.5, L - 0.5, ctx.n), r.uniform(0.5, L - 0.5, ctx.n),
                   r.uniform(-np.pi / 4, np.pi / 4, ctx.n), L)


@tool("construct", "Square lattice tilted by theta degrees (Erdos-Graham leaning), smallest box holding n.",
      theta=(10.0, 0.0, 45.0))
def lattice(ctx, state, theta=10.0):
    th = math.radians(theta)
    c, s = math.cos(th), math.sin(th)
    w = 0.5 * (abs(c) + abs(s))
    lo, hi = math.sqrt(ctx.n), math.ceil(math.sqrt(ctx.n)) + 2.0
    best = None
    for _ in range(30):  # bisect the box side; at each side try a few lattice phases
        L = 0.5 * (lo + hi)
        m = int(L * 2) + 3
        i, j = np.meshgrid(np.arange(-m, m + 1), np.arange(-m, m + 1))
        found = None
        for ph in ((0, 0), (0.5, 0), (0, 0.5), (0.5, 0.5), (0.25, 0.25)):
            u, v = i.ravel() + ph[0], j.ravel() + ph[1]
            px, py = L / 2 + u * c - v * s, L / 2 + u * s + v * c
            ok = (px >= w) & (px <= L - w) & (py >= w) & (py <= L - w)
            if ok.sum() >= ctx.n:
                found = (px[ok], py[ok])
                break
        if found:
            hi, best = L, (found, L)
        else:
            lo = L
    (px, py), L = best
    p = Packing(px, py, np.full(len(px), th), L)
    return fit_count(p, ctx.n, ctx.rng, "wall")


@tool("construct", "Friedman/Goebel tilted-block family: smallest (i + j sqrt2)/2 side whose block holds n.",
      budget=(20.0, 2.0, 120.0))
def tilted(ctx, state, budget=20.0):
    stop = min(ctx.deadline, time.time() + budget)
    top = best_store.best_side(ctx.n, ctx.store_root) + 0.5
    for s in candidate_sides(top, frac_min=0.05):
        if s * s < ctx.n * 0.95:
            continue
        for p, q, shift, order in configs(s):
            if time.time() > stop:
                return store(ctx, None)
            x, y, deg = tilted_block(s, p, q, shift, order)
            if len(x) >= ctx.n:
                pk = Packing(x, y, np.deg2rad(deg), s)
                # Touching is fine here: squeeze opens the margin.
                return fit_count(pk, ctx.n, ctx.rng, "wall")
    return store(ctx, None)


@tool("construct", "Stored packing of a larger m (n < m <= n + up) with m - n squares dropped.",
      up=(3, 1, 12), how=("wall", ["wall", "random", "crowded"]))
def larger(ctx, state, up=3, how="wall"):
    m = ctx.n + int(ctx.rng.integers(1, int(up) + 1))
    if m > best_store.MAX_N:
        return store(ctx, None)
    xyd = best_store.load_xyd(m, ctx.store_root)
    if xyd is None:
        return store(ctx, None)
    return _drop(from_xyd(*xyd), m - ctx.n, how, ctx.rng)


@tool("construct", "Stored packing of a smaller m wrapped in an L of axis squares (side + 1), extras dropped.")
def wrapped(ctx, state):
    lo = max(1, ctx.n - 2 * math.ceil(math.sqrt(ctx.n)) - 3)
    for m in range(lo, ctx.n):  # smallest m whose wrap still holds n
        if m + 2 * math.floor(best_store.best_side(m, ctx.store_root) + 1e-9) + 1 < ctx.n:
            continue
        xyd = best_store.load_xyd(m, ctx.store_root)
        base = from_xyd(*xyd) if xyd else grid(Ctx(m, ctx.rng), None)
        x, y, deg = l_wrap(base.x, base.y, np.rad2deg(base.t), base.L)
        return fit_count(Packing(x, y, np.deg2rad(deg), base.L + 1), ctx.n, ctx.rng, "random")
    return grid(ctx, None)


@tool("construct", "Random seed with C4 rotational symmetry about the box center (+ a 45-degree center square).",
      loose=(1.2, 1.05, 1.5))
def symmetric(ctx, state, loose=1.2):
    L = math.sqrt(ctx.n) * loose
    k, rem = divmod(ctx.n, 4)
    r = ctx.rng
    qx, qy = r.uniform(0, L / 2 - 0.5, k), r.uniform(0.5, L / 2, k)
    qt = r.uniform(-np.pi / 4, np.pi / 4, k)
    xs, ys, ts = [], [], []
    for q in range(4):  # rotate the quarter by q * 90 degrees around the center
        a = q * np.pi / 2
        dx, dy = qx - L / 2, qy - L / 2
        xs.append(L / 2 + dx * math.cos(a) - dy * math.sin(a))
        ys.append(L / 2 + dx * math.sin(a) + dy * math.cos(a))
        ts.append(qt + a)
    p = Packing(np.concatenate(xs), np.concatenate(ys), np.concatenate(ts), L)
    if rem:
        p = Packing(np.append(p.x, L / 2), np.append(p.y, L / 2), np.append(p.t, np.pi / 4), L)
        p = _add(p, rem - 1, r)
    return p


# ----------------------------------------------------------------------------- perturb


@tool("perturb", "Shake a random fraction of squares by Gaussian noise.",
      frac=(0.1, 0.01, 0.6), sigma=(0.2, 0.01, 0.8))
def jolt(ctx, state, frac=0.1, sigma=0.2):
    p = state.copy()
    k = max(1, int(ctx.rng.binomial(p.n, frac)))
    idx = ctx.rng.choice(p.n, k, replace=False)
    p.x[idx] += ctx.rng.normal(0, sigma, k)
    p.y[idx] += ctx.rng.normal(0, sigma, k)
    p.t[idx] += ctx.rng.normal(0, sigma, k)
    return p


@tool("perturb", "Pick k squares (random, near the top/right walls, or crowded) and move them into the biggest holes.",
      k=(1, 1, 8), how=("random", ["random", "wall", "crowded"]))
def relocate(ctx, state, k=1, how="random"):
    k = int(min(k, state.n - 1)) if state.n > 1 else 0
    return _add(_drop(state, k, how, ctx.rng), k, ctx.rng)


@tool("perturb", "Add a smooth random rotation field (low-frequency sines) to every square's angle.",
      amp=(0.15, 0.01, 0.8), freq=(1.0, 0.3, 3.0))
def twist(ctx, state, amp=0.15, freq=1.0):
    p = state.copy()
    a, b, ph = ctx.rng.normal(size=3)
    p.t += amp * np.sin(freq * 2 * np.pi * (a * p.x + b * p.y) / p.L + ph)
    return p


@tool("perturb", "Scale the layout into a box shrink times smaller (relaxers then have to resolve the pressure).",
      shrink=(0.995, 0.95, 0.9999))
def press(ctx, state, shrink=0.995):
    return Packing(state.x * shrink, state.y * shrink, state.t.copy(), state.L * shrink)


@tool("perturb", "Thermal noise on every square, sigma = temp (annealing kick).", temp=(0.05, 0.001, 0.3))
def heat(ctx, state, temp=0.05):
    p = state.copy()
    p.x += ctx.rng.normal(0, temp, p.n)
    p.y += ctx.rng.normal(0, temp, p.n)
    p.t += ctx.rng.normal(0, temp, p.n)
    return p


# ----------------------------------------------------------------------------- relax


@tool("relax", "FIRE minimization of the overlap energy at fixed box side.", maxiter=(3000, 200, 20000))
def relax(ctx, state, maxiter=3000):
    x, y, t, _ = relax_fire(state.x.copy(), state.y.copy(), state.t.copy(), state.L, MARGIN, int(maxiter))
    return Packing(x, y, t, state.L)


@tool("relax", "Penalty method with the side as a variable: min L + mu E for growing mu (nearly feasible).",
      mu_max=(1e5, 1e3, 1e7), maxiter=(400, 100, 3000))
def squash(ctx, state, mu_max=1e5, maxiter=400):
    mus = [m for m in (10.0, 100.0, 1e3, 1e4, 1e5, 1e6, 1e7) if m <= mu_max * 1.0001]
    x, y, t, L = compact(state.x, state.y, state.t, state.L, mus=tuple(mus), maxiter=int(maxiter))
    return Packing(x, y, t, L)


@tool("relax", "Shrink the box while the packing stays exactly feasible (grows it first if needed).",
      rounds=(30, 5, 200), step=(0.003, 1e-5, 0.03), method=("fire", ["fire", "lbfgs"]))
def squeeze_box(ctx, state, rounds=30, step=0.003, method="fire"):
    p, left = state, int(rounds)
    while left > 0:  # in chunks, so a slow squeeze still stops at the deadline
        r = squeeze(p.x, p.y, p.t, p.L, rounds=min(5, left), step=step, method=method,
                    maxiter=3000 if method == "fire" else 1500)
        if r is None:
            return p
        step = max(1e-7, step * 0.5) if r[3] >= p.L else step
        p, left = Packing(*r), left - 5
        if ctx.time_left() <= 0:
            break
    return p


@tool("relax", "Lifted (3D lens) compact: squares slide past each other in z while the box shrinks, then flatten.",
      maxiter=(300, 100, 2000))
def lift(ctx, state, maxiter=300):
    from ..lift import lift_compact
    x, y, t, L = lift_compact(state.x, state.y, state.t, state.L,
                              z=ctx.rng.uniform(-0.5, 0.5, state.n), maxiter=int(maxiter))
    return Packing(x, y, t, L)


@tool("relax", "Game-of-Life rule: every square moves only from its neighbours' state while the box shrinks "
      "(rule = an inventor's best evolved rule from the library, see wm.life_evolve).",
      rule=("open", ["granular", "crystal", "tension", "tilt", "spacer", "open"]), steps=(600, 100, 3000))
def life(ctx, state, rule="open", steps=600):
    from ..life import run_rule
    from ..life_evolve import library_rule
    x, y, t, L = run_rule(library_rule(rule), state.x, state.y, state.t, state.L, steps=int(steps),
                          seed=int(ctx.rng.integers(1 << 30)))
    return Packing(x, y, t, L)


# ----------------------------------------------------------------------------- combine


@tool("combine", "Cut along a random line through the center: keep state's squares on one side and the "
      "other recipe's on the other side, then fix the count.", other=([{"t": "store"}], None))
def crossover(ctx, state, other=None):
    b = run_steps(other or [{"t": "store"}], ctx, None)
    L = max(state.L, b.L)
    ka, kb = L / state.L, L / b.L
    ax, ay, bx, by = state.x * ka, state.y * ka, b.x * kb, b.y * kb
    ang = ctx.rng.uniform(0, np.pi)
    nx, ny = math.cos(ang), math.sin(ang)
    off = ctx.rng.uniform(-0.25, 0.25) * L
    sa = (ax - L / 2) * nx + (ay - L / 2) * ny - off
    sb = (bx - L / 2) * nx + (by - L / 2) * ny - off
    ia, ib = sa < 0, sb >= 0
    p = Packing(np.concatenate([ax[ia], bx[ib]]), np.concatenate([ay[ia], by[ib]]),
                np.concatenate([state.t[ia], b.t[ib]]), L)
    return fit_count(p, ctx.n, ctx.rng, "crowded")


# ----------------------------------------------------------------------------- meta


@tool("meta", "Basin hopping: run body times (0 = until the time budget ends); keep a feasible result if "
      "smaller, or worse with Metropolis probability exp(-dL / temp).", times=(5, 0, 50), temp=(0.0, 0.0, 0.01),
      body=([], None))
def repeat(ctx, state, times=5, temp=0.0, body=None):
    cur = finish(state, ctx)
    best = cur
    k = 0
    while k < int(times) or (int(times) <= 0 and k < 100000):
        k += 1
        if ctx.time_left() <= 0:
            break
        cand = finish(run_steps(body or [], ctx, cur.copy()), ctx)
        if not is_feasible(cand):
            continue
        dL = cand.L - cur.L
        if dL < -1e-12 or (temp > 0 and ctx.rng.random() < math.exp(-dL / temp)):
            cur = cand
        if cur.L < best.L - 1e-12 and is_feasible(cur):
            best = cur
    return best


@tool("meta", "Run body k times from the same state, keep the smallest feasible result.",
      k=(3, 1, 20), body=([], None))
def best_of(ctx, state, k=3, body=None):
    best = None
    for _ in range(int(k)):
        if best is not None and ctx.time_left() <= 0:
            break
        cand = finish(run_steps(body or [], ctx, state.copy() if state else None), ctx)
        if is_feasible(cand) and (best is None or cand.L < best.L):
            best = cand
    return best if best is not None else finish(state, ctx)


# ----------------------------------------------------------------------------- running recipes


def run_steps(steps, ctx: Ctx, state: Packing | None) -> Packing:
    for step in steps:
        args = {k: v for k, v in step.items() if k != "t"}
        name = step["t"]
        if name not in TOOLS:
            raise KeyError(f"unknown tool {name!r}")
        if state is None and TOOLS[name]["kind"] != "construct":
            state = store(ctx, None)
        state = TOOLS[name]["fn"](ctx, state, **args)
        if state.n != ctx.n:
            state = fit_count(state, ctx.n, ctx.rng)
    return state if state is not None else store(ctx, None)


def finish(p: Packing, ctx: Ctx) -> Packing:
    """Make a packing exactly feasible: squeeze it (which first grows the box until it relaxes)."""
    if p is None:
        return store(ctx, None)
    if is_feasible(p):
        return p
    r = squeeze(p.x, p.y, p.t, p.L, rounds=40, step=0.001)
    return Packing(*r) if r is not None else store(ctx, None)


def recipe_id(recipe) -> str:
    return hashlib.sha1(json.dumps(recipe, sort_keys=True).encode()).hexdigest()[:10]


def run_recipe(recipe, n: int, seed: int = 0, budget: float = 120.0, store_root=None) -> Packing:
    ctx = Ctx(n, np.random.default_rng(seed), time.time() + budget, store_root)
    return finish(run_steps(recipe, ctx, None), ctx)


def evaluate(recipe, n: int, seed: int = 0, budget: float = 120.0, source: str = "swarm",
             store_root=None, offer: bool = True, restarts: bool = True) -> dict:
    """Run a recipe on n (again with new seeds while budget remains, if `restarts`), and offer the
    best result to the store. Returns a result record for the board."""
    before = best_store.best_side(n, store_root)
    t0 = time.time()
    stop = t0 + budget
    best, err, tries, sides = None, None, 0, []
    while True:
        tries += 1
        try:
            p = run_recipe(recipe, n, seed + tries - 1, stop - time.time(), store_root)
        except Exception as e:  # a bad recipe is a result too: the board should learn it fails
            err = f"{type(e).__name__}: {e}"
            break
        if is_feasible(p) and (best is None or p.L < best.L):
            best = p
        sides.append(round(p.L, 12))
        # Stop when out of time, or when the recipe is deterministic (same side twice running).
        if not restarts or time.time() > stop - 0.1 * budget or (len(sides) >= 2 and sides[-1] == sides[-2]):
            break
    res = {"n": n, "recipe_id": recipe_id(recipe), "seed": seed, "tries": tries, "before": before,
           "side": None if best is None else float(best.L), "accepted": False,
           "seconds": round(time.time() - t0, 2), "error": err}
    if best is not None and best.L < before - 1e-9 and offer:
        r = best_store.offer(n, best.x, best.y, np.rad2deg(best.t), source, root=store_root)
        res["accepted"] = r["accepted"]
        res["reason"] = r["reason"]
        if r["side"]:
            res["side"] = r["side"]
    res["gain"] = (puzzle_score(before, n) - puzzle_score(res["side"], n)) if res["accepted"] else 0.0
    return res


def describe() -> str:
    """Markdown table of every tool, for agents (and people) choosing what to compose."""
    lines = ["| tool | kind | params | what it does |", "|---|---|---|---|"]
    for name, t in TOOLS.items():
        ps = ", ".join(f"{k}={v[0]!r}" if not isinstance(v[0], list) else k for k, v in t["params"].items())
        lines.append(f"| {name} | {t['kind']} | {ps} | {t['doc']} |")
    return "\n".join(lines)
