"""Agent personas for the packing swarm, each with its own theoretical grounding.

An agent loops: read the board, pick a puzzle size n (inside its band, weighted toward sizes
whose stored packing looks slack next to its neighbours, avoiding sizes other agents have
claimed), propose a recipe, claim n, run the recipe, offer the result to the store, post the
result. Proposals come from three places:

* its own grammar: the moves its theory suggests (a physicist compresses slowly and anneals,
  a combinatorialist builds from known families, ...)
* borrowing: take a recipe another agent posted that did well and splice in its own signature
  move, citing the parent so the board records the lineage
* mutation: nudge the parameters or swap a step of a recipe it already likes

New recipes are posted with the reasoning behind them, and some come with a hypothesis
("this recipe improves n in a-b") that the board later scores from results.
"""

import copy
import json
import math
import random as pyrandom
import time
from dataclasses import dataclass, field

import numpy as np

from .. import best_store
from .board import Board
from .tools import TOOLS, evaluate, recipe_id


# ----------------------------------------------------------------------------- recipe algebra


def _steps(recipe):
    """Every step of a recipe, nested ones included (bodies of meta steps, `other` of combine steps)."""
    for s in recipe:
        yield s
        for key in ("body", "other"):
            if isinstance(s.get(key), list):
                yield from _steps(s[key])


def mutate(recipe, rng: pyrandom.Random, rate: float = 0.5):
    """Copy of the recipe with one or more numeric/categorical parameters resampled near their value."""
    r = copy.deepcopy(recipe)
    steps = [s for s in _steps(r) if TOOLS[s["t"]]["params"]]
    if not steps:
        return r
    for s in rng.sample(steps, max(1, int(len(steps) * rate))):
        name, spec = rng.choice(list(TOOLS[s["t"]]["params"].items()))
        if isinstance(spec[0], list):
            continue
        if len(spec) == 2 and isinstance(spec[1], list):
            s[name] = rng.choice(spec[1])
            continue
        default, lo, hi = spec
        cur = s.get(name, default)
        if isinstance(default, int):
            s[name] = int(min(hi, max(lo, cur + rng.choice([-1, 1]) * max(1, abs(cur) // 3))))
        elif lo > 0 and hi / lo > 50:  # log-scale parameter
            s[name] = float(min(hi, max(lo, cur * math.exp(rng.gauss(0, 0.7)))))
        else:
            s[name] = float(min(hi, max(lo, cur + rng.gauss(0, 0.25) * (hi - lo))))
    return r


def splice(a, b, rng: pyrandom.Random):
    """Head of recipe a (up to a random cut) followed by the tail of b."""
    i = rng.randint(1, max(1, len(a)))
    j = rng.randint(0, max(0, len(b) - 1))
    tail = b[j:]
    if any(TOOLS[s["t"]]["kind"] == "construct" for s in a[:i]):
        while tail and TOOLS[tail[0]["t"]]["kind"] == "construct":  # don't throw away a's packing
            tail = tail[1:]
    return copy.deepcopy(a[:i] + tail)[:8]


def summarize(recipe) -> str:
    parts = []
    for s in recipe:
        args = ",".join(f"{k}={v:g}" if isinstance(v, (int, float)) else f"{k}={v}"
                        for k, v in s.items() if k not in ("t", "body", "other"))
        inner = ""
        if "body" in s:
            inner = "[" + summarize(s["body"]) + "]"
        if "other" in s:
            inner = "<" + summarize(s["other"]) + ">"
        parts.append(f"{s['t']}{'(' + args + ')' if args else ''}{inner}")
    return " > ".join(parts)


# ----------------------------------------------------------------------------- where to look


class Opportunities:
    """Which n look slack: stored side vs the side implied by the best density nearby.

    Packing density n / L^2 varies smoothly with n, so a size whose density sits well below its
    neighbours' best probably has room. Refreshed from the store every few minutes.
    """

    def __init__(self, store_root=None, window: int = 6, ttl: float = 300.0):
        self.store_root, self.window, self.ttl = store_root, window, ttl
        self.t = 0.0
        self.side = {}

    def refresh(self):
        if time.time() - self.t < self.ttl and self.side:
            return
        self.side = {n: best_store.best_side(n, self.store_root) for n in range(1, best_store.MAX_N + 1)}
        self.t = time.time()

    def gap(self, n: int) -> float:
        self.refresh()
        dens = lambda m: m / self.side[m] ** 2
        nb = [dens(m) for m in range(max(2, n - self.window), min(best_store.MAX_N, n + self.window) + 1)]
        target = math.sqrt(n / max(nb))
        return max(0.0, (self.side[n] - target) / math.sqrt(n))


# ----------------------------------------------------------------------------- personas


@dataclass
class Persona:
    name: str
    school: str
    grounding: str
    band: tuple  # (lo, hi) puzzle sizes it prefers
    borrow: float  # probability of building on another agent's recipe
    own: callable = None  # rng, n -> (recipe, why)
    hypothesis: str = ""  # template for hypotheses about its own recipes
    collaborative: bool = False  # joins claimed n instead of avoiding them


def _erdos(rng, n):
    seed = rng.choice([{"t": "tilted", "budget": 15.0}, {"t": "larger", "up": rng.randint(1, 6),
                       "how": rng.choice(["wall", "crowded", "random"])},
                       {"t": "wrapped"}, {"t": "lattice", "theta": rng.uniform(2, 40)},
                       {"t": "compose", "rank": rng.randint(0, 2)}, {"t": "compose", "rank": 0}])
    r = [seed, {"t": "squash", "mu_max": 1e5}, {"t": "squeeze_box", "rounds": 30, "step": 0.002}]
    if rng.random() < 0.5:
        r.append({"t": "repeat", "times": 4, "body": [{"t": "relocate", "k": 1, "how": "wall"},
                                                      {"t": "squeeze_box", "rounds": 20, "step": 0.001}]})
    return r, (f"Monotonicity and known families: build n from a structured packing ({seed['t']}), "
               "then let the optimizer close the slack the family leaves.")


def _jammer(rng, n):
    seed = {"t": "random", "loose": rng.uniform(1.15, 1.45)}
    r = [seed, {"t": "squash", "mu_max": 10 ** rng.uniform(4, 6)},
         {"t": "repeat", "times": rng.randint(3, 8), "temp": 10 ** rng.uniform(-4, -2.5),
          "body": [{"t": "heat", "temp": rng.uniform(0.01, 0.08)}, {"t": "squeeze_box", "rounds": 25}]}]
    if rng.random() < 0.35:
        r.insert(1, {"t": "lift", "maxiter": 300})
    return [{"t": "best_of", "k": rng.randint(2, 5), "body": r}], (
        "Jamming: slow compression of a loose granular gas finds dense jammed states; thermal kicks with "
        "Metropolis acceptance let it escape shallow jams (and a lift into 3D lets grains pass).")


def _darwin(rng, n):
    other = rng.choice([[{"t": "store"}, {"t": "jolt", "frac": 0.3, "sigma": 0.3}],
                        [{"t": "larger", "up": 2}], [{"t": "random", "loose": 1.3}, {"t": "squash"}],
                        [{"t": "symmetric"}, {"t": "squash"}]])
    r = [{"t": "store"}, {"t": "best_of", "k": rng.randint(3, 6), "body": [
        {"t": "crossover", "other": other}, {"t": "squash", "mu_max": 1e5}, {"t": "squeeze_box", "rounds": 30}]}]
    return r, ("Recombination: a child that keeps one half of the champion and takes the other half from a "
               "different lineage can land in a basin neither parent reaches by mutation.")


def _lagrange(rng, n):
    kick = rng.choice([{"t": "press", "shrink": 1 - 10 ** rng.uniform(-4, -2.3)},
                       {"t": "breathe", "frac": rng.uniform(0.1, 0.5), "s_min": rng.uniform(0.6, 0.9)}])
    r = [{"t": "store"}, kick,
         {"t": "squash", "mu_max": 10 ** rng.uniform(5, 7), "maxiter": rng.choice([400, 1000, 2000])},
         {"t": "squeeze_box", "rounds": 60, "step": 1e-4, "method": rng.choice(["lbfgs", "fire"])}]
    return r, ("Constrained optimization: the stored packing is a KKT point of a relaxed problem only if every "
               "contact is active; pressing the box and re-solving with a stiffer penalty finds the slack.")


def _metropolis(rng, n):
    move = rng.choice([{"t": "jolt", "frac": rng.uniform(0.03, 0.2), "sigma": rng.uniform(0.05, 0.4)},
                       {"t": "relocate", "k": rng.randint(1, 3), "how": rng.choice(["random", "crowded", "wall"])},
                       {"t": "twist", "amp": rng.uniform(0.03, 0.3)},
                       {"t": "breathe", "frac": rng.uniform(0.1, 0.4), "s_min": rng.uniform(0.5, 0.9)}])
    r = [{"t": "store"}, {"t": "repeat", "times": 0, "temp": rng.choice([0.0, 1e-4, 5e-4]),
                          "body": [move, {"t": "squeeze_box", "rounds": 25, "step": 0.001}]}]
    return r, (f"Basin hopping: a {move['t']} move followed by a local squeeze samples neighbouring jammed "
               "states; with a small temperature the walk can cross ridges between basins.")


def _homotopy(rng, n):
    seed = rng.choice([{"t": "morph_remove", "steps": rng.randint(4, 12), "how": rng.choice(["wall", "random"])},
                       {"t": "morph_insert", "steps": rng.randint(4, 12), "hole": rng.randint(0, 3)},
                       {"t": "morph_remove", "fractional": "no", "how": "wall"}])
    r = [{"t": "best_of", "k": rng.randint(2, 4), "body": [seed, {"t": "squeeze_box", "rounds": 30, "step": 0.0005}]}]
    if rng.random() < 0.5:
        r.append({"t": "repeat", "times": 0, "body": [
            {"t": "breathe", "frac": rng.uniform(0.1, 0.4), "s_min": rng.uniform(0.5, 0.9)},
            {"t": "squeeze_box", "rounds": 20, "step": 0.0005}]})
    return r, ("Continuation: treat n as continuous (one square of fractional size) and follow the optimum from "
               "a neighbouring n, so the packing deforms instead of being rebuilt.")


def _escher(rng, n):
    r = [{"t": "best_of", "k": rng.randint(3, 8), "body": [
        {"t": "symmetric", "loose": rng.uniform(1.1, 1.35)}, {"t": "squash", "mu_max": 1e5},
        {"t": "squeeze_box", "rounds": 30}]}]
    if rng.random() < 0.5:
        r.append({"t": "repeat", "times": 5, "body": [{"t": "twist", "amp": 0.1}, {"t": "squeeze_box", "rounds": 20}]})
    return r, ("Symmetry: many best-known packings (n = 10, 11, 18, 19, ...) are near-symmetric under a quarter "
               "turn; seeding inside that subgroup shrinks the search space by four.")


def _conway(rng, n):
    rule = rng.choice(["granular", "crystal", "tension", "tilt", "spacer", "open"])
    seed = rng.choice([{"t": "store"}, {"t": "random", "loose": rng.uniform(1.15, 1.4)}, {"t": "larger", "up": 1}])
    r = [seed, {"t": "life", "rule": rule, "steps": rng.choice([400, 800, 1600])},
         {"t": "squash", "mu_max": 1e5}, {"t": "squeeze_box", "rounds": 30, "step": 0.002}]
    return r, (f"Local rules: let every square react only to its neighbours under the evolved '{rule}' rule "
               "while the walls close in; structure that emerges this way works at any n.")


PERSONAS = {
    "erdos": Persona("erdos", "combinatorial constructions",
                     "Erdos-Graham leaning strips, Goebel and Friedman families, monotonicity in n "
                     "(drop squares from n+k, wrap n-k in an L)", (20, 1000), 0.25, _erdos,
                     "Construct-then-squeeze with {summary} beats the store for n in {lo}-{hi}"),
    "jammer": Persona("jammer", "statistical physics",
                      "granular jamming, simulated annealing, Lubachevsky-Stillinger compression, "
                      "lifting into an extra dimension", (5, 120), 0.2, _jammer,
                      "Annealed compression ({summary}) finds denser jammed states for n in {lo}-{hi}"),
    "darwin": Persona("darwin", "evolutionary computation",
                      "genetic algorithms: crossover between lineages, tournament selection, and "
                      "evolving the recipes themselves", (10, 250), 0.5, _darwin,
                      "Crossover ({summary}) escapes the champion's basin for n in {lo}-{hi}"),
    "lagrange": Persona("lagrange", "continuous optimization",
                        "penalty and barrier methods, KKT conditions, L-BFGS; shaving slack off "
                        "stored packings", (10, 1000), 0.2, _lagrange,
                        "Stored packings for n in {lo}-{hi} still have slack that {summary} removes"),
    "metropolis": Persona("metropolis", "stochastic local search",
                          "basin hopping, Monte Carlo moves, tabu-like relocation of rattlers "
                          "into holes", (10, 300), 0.3, _metropolis,
                          "Basin hopping ({summary}) improves n in {lo}-{hi}"),
    "homotopy": Persona("homotopy", "numerical continuation",
                        "homotopy in n: grow or shrink one square continuously from the n-1 / n+1 optimum, "
                        "breathing patches of squares", (10, 400), 0.2, _homotopy,
                        "Continuation from a neighbouring n ({summary}) beats the store for n in {lo}-{hi}"),
    "escher": Persona("escher", "symmetry and group theory",
                      "packings invariant under C4 or mirror subgroups; symmetric seeds and "
                      "twist fields", (5, 80), 0.2, _escher,
                      "C4-symmetric seeds ({summary}) reach new basins for n in {lo}-{hi}"),
    "conway": Persona("conway", "cellular automata",
                      "Game-of-Life style local rules: each square updates from its neighbours' contacts, "
                      "gaps and angles only; rules are evolved across many n (wm.life_evolve)",
                      (5, 400), 0.2, _conway,
                      "Neighbour rules ({summary}) reach denser packings for n in {lo}-{hi}"),
    "bandit": Persona("bandit", "reinforcement learning",
                      "multi-armed bandits: Thompson sampling over the board's recipes by n band, "
                      "splicing the best arms together", (5, 400), 1.0, None,
                      "Splicing top recipes ({summary}) improves n in {lo}-{hi}", collaborative=True),
}


# ----------------------------------------------------------------------------- the agent


@dataclass
class Agent:
    name: str
    persona: Persona
    board: Board
    seed: int = 0
    budget: float = 90.0  # seconds per run
    max_n: int = 1000
    store_root: object = None
    opp: Opportunities = None
    rng: pyrandom.Random = field(init=False)

    def __post_init__(self):
        self.rng = pyrandom.Random(self.seed)
        self.opp = self.opp or Opportunities(self.store_root)

    # -- choosing n
    def choose_n(self) -> int:
        lo, hi = self.persona.band
        hi = min(hi, self.max_n)
        claimed = {} if self.persona.collaborative else self.board.active_claims(exclude_agent=self.name)
        cands = [n for n in range(lo, hi + 1) if n not in claimed]
        if not cands:
            return self.rng.randint(lo, hi)
        cands = self.rng.sample(cands, min(60, len(cands)))
        w = [1e-4 + self.opp.gap(n) for n in cands]
        return self.rng.choices(cands, weights=w)[0]

    # -- proposing a recipe
    def _board_pick(self, n: int):
        """A promising recipe from the board for size n, Thompson-sampled on win rate and side change."""
        stats = self.board.recipe_stats()
        recipes = self.board.recipes()
        best, best_v = None, -math.inf
        for rid, rp in recipes.items():
            if rp["agent"] == self.name and self.persona.name != "bandit":
                continue
            s = stats.get(rid, {"runs": 0, "wins": 0, "rel": 0.0, "errors": 0, "ns": set()})
            if s["errors"] > s["runs"] // 2 and s["runs"] >= 2:
                continue
            near = any(abs(m - n) <= max(5, n // 5) for m in s.get("ns", ()))
            v = self.rng.betavariate(1 + s["wins"], 1 + s["runs"] - s["wins"] + 1)
            v -= 2.0 * max(-0.05, min(0.05, s["rel"] if s["rel"] != math.inf else 0.0))
            v += 0.05 if near else 0.0
            if v > best_v:
                best, best_v = rp, v
        return best

    def propose(self, n: int):
        p = self.persona
        if self.rng.random() < 0.2:  # retest a recipe that has won somewhere, so hypotheses get evidence
            stats = self.board.recipe_stats()
            winners = [rid for rid, s in stats.items() if s["wins"] and n not in s["win_ns"]]
            recipes = self.board.recipes()
            winners = [rid for rid in winners if rid in recipes]
            if winners:
                rid = self.rng.choice(winners)
                return recipes[rid]["recipe"], [rid], f"Retest {recipes[rid]['agent']}'s winning recipe at a new n."
        if p.own is None or self.rng.random() < p.borrow:
            parent = self._board_pick(n)
            if parent is not None:
                if p.name == "bandit":
                    other = self._board_pick(n)
                    if other is not None and other["recipe_id"] != parent["recipe_id"] and self.rng.random() < 0.5:
                        return (splice(parent["recipe"], other["recipe"], self.rng),
                                [parent["recipe_id"], other["recipe_id"]],
                                f"Splice of two arms with high posterior win rate ({parent['agent']}, {other['agent']}).")
                    return parent["recipe"], [parent["recipe_id"]], "Exploit: highest Thompson sample on the board."
                if p.name == "darwin" and self.rng.random() < 0.5:
                    return mutate(parent["recipe"], self.rng), [parent["recipe_id"]], (
                        f"Mutation of {parent['agent']}'s recipe: inherit what works, vary the parameters.")
                mine, why = p.own(self.rng, n)
                return (splice(parent["recipe"], mine, self.rng), [parent["recipe_id"]],
                        f"Built on {parent['agent']}'s recipe with my own tail. {why}")
        if p.own is None:  # bandit with an empty board: start from a plain baseline
            return [{"t": "store"}, {"t": "jolt"}, {"t": "squeeze_box"}], [], "Baseline arm."
        r, why = p.own(self.rng, n)
        return r, [], why

    # -- one turn
    def step(self) -> dict:
        n = self.choose_n()
        recipe, parents, why = self.propose(n)
        rid = recipe_id(recipe)
        known = self.board.recipes()
        if rid not in known:
            self.board.post(self.name, "recipe", summarize(recipe), self.persona.name, recipe=recipe,
                            recipe_id=rid, parents=parents, why=why)
            if self.rng.random() < 0.3:
                lo, hi = max(1, n - 15), min(self.max_n, n + 15)
                self.board.post(self.name, "hypothesis", self.persona.hypothesis.format(
                    summary=summarize(recipe)[:80], lo=lo, hi=hi), self.persona.name, recipe_id=rid, band=[lo, hi])
        self.board.claim(self.name, n, self.budget / 60 * 1.5, self.persona.name)
        res = evaluate(recipe, n, seed=self.rng.randrange(1 << 30), budget=self.budget,
                       source=f"swarm:{self.persona.name}:{rid}", store_root=self.store_root)
        self.board.post(self.name, "result", f"n={n} {res['before']:.6f} -> {res['side']}", self.persona.name, **res)
        if res["accepted"]:
            self.board.post(self.name, "note", f"store win at n={n}: {res['before']:.6f} -> {res['side']:.6f} "
                            f"with {summarize(recipe)[:120]}", self.persona.name, recipe_id=rid)
            self.opp.t = 0.0
        return res

    def run(self, minutes: float, log=print):
        stop = time.time() + 60 * minutes
        while time.time() < stop:
            r = self.step()
            log(f"[{self.name}] n={r['n']} {r['before']:.5f} -> {r['side']} "
                f"{'WIN' if r['accepted'] else ''} {r['seconds']}s {r.get('error') or ''}")


def run_agent(args):
    """Process entry point: (name, persona, board_dir, seed, minutes, budget, max_n)."""
    name, persona, board_dir, seed, minutes, budget, max_n = args
    import os
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    np.random.seed(seed)
    a = Agent(name, PERSONAS[persona], Board(board_dir), seed=seed, budget=budget, max_n=max_n)
    log_path = Board(board_dir).root / "logs" / f"{name}.log"
    log_path.parent.mkdir(exist_ok=True)
    with open(log_path, "a") as f:
        def log(msg):
            f.write(time.strftime("%H:%M:%S ") + msg + "\n")
            f.flush()
        try:
            a.run(minutes, log)
        except Exception as e:
            log(f"crashed: {type(e).__name__}: {e}")
            raise


def persona_prompt(name: str) -> str:
    """Instructions for a Claude session acting as this persona on the board (instead of the
    built-in heuristic agent): what to read, what tools exist, how to post."""
    from .tools import describe
    p = PERSONAS[name]
    return f"""You are agent `{name}` in a research swarm packing n unit squares into the smallest square
(Kaggle Squares in Squares; score sum of L/sqrt(n) - 1). Your school: {p.school}. Grounding: {p.grounding}.
Prefer n in {p.band[0]}-{p.band[1]}.

Loop: read the board digest (`python -m scripts.swarm digest`), pick an n others have not claimed,
write a recipe (JSON list of tool steps, see below) that your theory says should work, possibly
building on another agent's recipe (cite it with --parents), and run it:

    python -m scripts.swarm try --agent {name} --persona {name} --n <n> --recipe '<json>' \\
        --why '<your reasoning>' [--parents <recipe ids>] [--hypothesis '<claim>' --band lo-hi]

That claims n, runs the recipe, offers the result to the store and posts recipe, result and
hypothesis to the board. Post free-form observations with `python -m scripts.swarm note`.

Tools:
{describe()}
"""


def describe_personas() -> str:
    return "\n".join(f"- **{p.name}** ({p.school}, n {p.band[0]}-{p.band[1]}): {p.grounding}"
                     for p in PERSONAS.values())
