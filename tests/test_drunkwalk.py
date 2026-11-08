import numpy as np

from wm import drunkwalk
from wm.pack_opt import feasible


def test_lines_stay_ordered_with_straight_walls():
    rng = np.random.default_rng(0)
    lines = drunkwalk.drunk_lines(5, 5.0, 1.0, rng, 21)
    assert np.all(lines[0] == 0.0) and np.all(lines[-1] == 5.0)
    assert np.all(np.diff(lines, axis=0) >= 0)


def test_zero_sigma_is_the_plain_grid():
    rng = np.random.default_rng(0)
    x, y, t = drunkwalk.drunk_grid(9, 3, 3, 3.0, 0.0, rng)
    assert np.allclose(sorted(x), sorted(np.repeat([0.5, 1.5, 2.5], 3)))
    assert np.allclose(t, 0.0)


def test_trial_returns_feasible_packing():
    side, x, y, t = drunkwalk.trial(10, 3, 4, 0.6, seed=1)
    assert len(x) == 10 and side <= 4.0 + 1e-9
    assert feasible(x, y, t, side)
