"""Checks for the Squares in Squares world. Run with `python -m pytest tests`."""

import numpy as np

from wm.squares import (KNOWN_BEST, NUM_ACTIONS, SQUEEZE, SquarePackingEnv, SquaresConfig, enclosing_side,
                        five_solution, grid_solution, pair_overlaps, polish, relax, validate)


def test_grid_is_valid():
    for n in range(1, 30):
        r = validate(*grid_solution(n))
        assert r["valid"]
        assert np.isclose(r["side"], np.ceil(np.sqrt(n)))


def test_five_optimum_touches_but_does_not_overlap():
    r = validate(*five_solution())
    assert r["valid"]
    assert np.isclose(r["side"], KNOWN_BEST[5])
    x, y, deg = five_solution()
    deg[4] = 40.0  # turning the middle square away from 45 degrees makes it hit the corners
    assert not validate(x, y, deg)["valid"]


def test_overlap_depth_and_normal():
    i, j, depth, normal = pair_overlaps([0.0, 0.7], [0.0, 0.0], [0.0, 0.0])
    assert list(i) == [0] and list(j) == [1]
    assert np.isclose(depth[0], 0.3)
    assert np.allclose(normal[0], [1, 0])
    # A diamond next to a square: the corner pokes in by 0.5 + sqrt(2)/2 - 1.1.
    _, _, depth, _ = pair_overlaps([0.0, 1.1], [0.0, 0.0], [0.0, 45.0])
    assert np.isclose(depth[0], 0.5 + np.sqrt(2) / 2 - 1.1)


def test_enclosing_side_of_rotated_square():
    assert np.isclose(enclosing_side([0.0], [0.0], [45.0]), np.sqrt(2))


def test_relax_separates_and_contains():
    x, y, deg = relax([1.0, 1.3, 1.6], [1.0, 1.1, 0.9], [0.0, 10.0, -20.0], side=4.0)
    for _ in range(20):
        x, y, deg = relax(x, y, deg, side=4.0)
    assert validate(x, y, deg, tol=1e-3)["valid"]
    _, _, _, report = polish(x, y, deg)
    assert report["valid"]


def test_env_step_shapes_and_reward():
    env = SquarePackingEnv(SquaresConfig(n=4, max_steps=50), seed=0)
    obs = env.reset()
    assert obs.shape == (64, 64) and obs.dtype == np.float32
    assert env.state().shape == (5, 3)
    total, start = 0.0, env.best
    for t in range(50):
        obs, r, done, info = env.step(SQUEEZE if t % 2 == 0 else int(t % NUM_ACTIONS))
        assert r >= 0
        total += r
    assert done
    assert np.isclose(total, start - env.best)
    assert 0.0 <= obs.min() and obs.max() <= 1.0


def test_submission_rows_use_radians():
    from wm.squares import SUBMISSION_COLUMNS, to_submission_rows
    rows = to_submission_rows({5: five_solution()})
    assert len(rows) == 1000 * 1001 // 2 and list(rows[0]) == SUBMISSION_COLUMNS
    assert [r["id"] for r in rows[:3]] == [0, 1, 2]
    five = [r for r in rows if r["n"] == 5]
    assert [r["square_id"] for r in five] == list(range(5)) and five[0]["id"] == 10
    assert np.isclose(five[4]["angle"], np.pi / 4)
    # Unsolved puzzles fall back to the grid, which is what the sample submission contains.
    assert [(r["x"], r["y"]) for r in rows if r["n"] == 3] == [(0.5, 0.5), (1.5, 0.5), (0.5, 1.5)]


def test_polished_packing_passes_official_metric():
    from wm.squares import official_check, to_submission_rows
    # The exact n = 5 optimum touches, which the official interval check cannot certify.
    exact = to_submission_rows({5: five_solution()}, fill_to=None)
    assert not official_check(exact)["valid"]
    x, y, deg, report = polish(*five_solution())
    assert report["valid"] and report["worst_overlap"] <= -1e-9
    c = official_check(to_submission_rows({5: (x, y, deg)}, fill_to=None))
    assert c["valid"], c["reason"]
    assert abs(c["side"] - KNOWN_BEST[5]) < 1e-7
    # The grid fallback is accepted as is.
    assert official_check(to_submission_rows({}, fill_to=9)[-9:])["valid"]
