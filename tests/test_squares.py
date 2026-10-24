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
