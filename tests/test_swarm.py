"""Checks for the swarm: every tool runs, recipes compose and evaluate, the board round-trips,
and an agent can take a full turn against a private store and board."""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from wm import best_store
from wm.swarm import agents, tools
from wm.swarm.board import Board
from wm.swarm.tools import TOOLS, Ctx, evaluate, is_feasible, recipe_id, run_recipe, run_steps

REPO_SOLUTIONS = Path(__file__).resolve().parent.parent / "solutions"


@pytest.fixture
def store(tmp_path):
    root = tmp_path / "store"
    shutil.copytree(REPO_SOLUTIONS, root)
    return root


def _example(name):
    kind = TOOLS[name]["kind"]
    if kind == "construct":
        return [{"t": name}]
    if kind == "combine":
        return [{"t": "store"}, {"t": name, "other": [{"t": "random"}]}]
    if kind == "meta":
        return [{"t": "store"}, {"t": name, "body": [{"t": "jolt"}, {"t": "squeeze_box", "rounds": 5}]}]
    return [{"t": "store"}, {"t": name}]


@pytest.mark.parametrize("name", sorted(TOOLS))
def test_every_tool_gives_n_squares_and_a_feasible_finish(name, store):
    for n in (5, 11):
        p = run_recipe(_example(name), n, seed=1, budget=10, store_root=store)
        assert p.n == n
        assert is_feasible(p)
        assert p.L <= np.ceil(np.sqrt(n)) * 1.6


def test_recipe_id_is_stable_and_order_insensitive_in_args():
    a = [{"t": "jolt", "frac": 0.1, "sigma": 0.2}]
    b = [{"sigma": 0.2, "t": "jolt", "frac": 0.1}]
    assert recipe_id(a) == recipe_id(b)


def test_evaluate_offers_only_improvements(store):
    # n = 6 has nothing stored: anything beating the 3x3 grid side is a win. The known
    # optimum is 3 (no improvement possible), so a squeeze from random must not be accepted.
    res = evaluate([{"t": "random"}, {"t": "squash"}, {"t": "squeeze_box"}], 6, budget=5, store_root=store)
    assert res["before"] == 3.0 and not res["accepted"]
    # n = 5 has no stored packing in a fresh dir: 2 + 1/sqrt2 beats the grid side 3.
    empty = store.parent / "empty"
    res = evaluate([{"t": "random", "loose": 1.3}, {"t": "squash"}, {"t": "squeeze_box", "rounds": 40}], 5,
                   budget=20, store_root=empty)
    assert res["accepted"] and res["side"] < 3.0
    assert best_store.best_side(5, empty) == pytest.approx(res["side"])


def test_bad_recipe_is_reported_not_raised(store):
    res = evaluate([{"t": "no_such_tool"}], 5, budget=2, store_root=store)
    assert res["error"] and not res["accepted"]


def test_mutate_and_splice_produce_runnable_recipes(store):
    import random
    rng = random.Random(0)
    for p in agents.PERSONAS.values():
        if p.own is None:
            continue
        r, why = p.own(rng, 12)
        assert why
        for child in (agents.mutate(r, rng), agents.splice(r, [{"t": "jolt"}, {"t": "squeeze_box"}], rng)):
            json.dumps(child)
            ctx = Ctx(12, np.random.default_rng(0), deadline=__import__("time").time() + 3, store_root=store)
            assert run_steps(child, ctx, None).n == 12


def test_board_round_trip_and_digest(tmp_path):
    b = Board(tmp_path / "board")
    b.post("a", "recipe", "x", recipe=[{"t": "store"}], recipe_id="r1", parents=[])
    b.post("a", "hypothesis", "r1 helps", recipe_id="r1", band=[5, 9])
    b.claim("a", 7, 5)
    b.post("a", "result", "", n=7, recipe_id="r1", before=3.0, side=2.9, accepted=True, gain=0.03, seconds=1)
    b2 = Board(tmp_path / "board")  # a fresh reader sees everything
    assert set(b2.recipes()) == {"r1"}
    assert b2.active_claims() == {7: ["a"]} and b2.active_claims(exclude_agent="a") == {}
    assert b2.recipe_stats()["r1"]["wins"] == 1
    assert b2.hypothesis_status(b2.posts("hypothesis")[0]) == "supported"
    assert "r1" in b2.digest()


def test_agent_turn_posts_recipe_claim_and_result(store, tmp_path):
    b = Board(tmp_path / "board")
    for i, name in enumerate(agents.PERSONAS):
        a = agents.Agent(name, agents.PERSONAS[name], b, seed=i, budget=3, max_n=30, store_root=store)
        res = a.step()
        assert res["error"] is None, res
    kinds = {p["kind"] for p in b.posts()}
    assert {"recipe", "claim", "result"} <= kinds
    assert len(b.posts("result")) == len(agents.PERSONAS)
