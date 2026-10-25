import numpy as np

from wm.pack_opt import energy_grad
from wm.sieve import _plate, bars_energy_grad, close_box, settle, simulate


def test_bars_gradient_matches_finite_differences():
    rng = np.random.default_rng(1)
    bars = np.array([[0.0, 0.0, 1.3, 0.2], [2.0, 1.0, 0.3, 0.9]])
    checked = 0
    for _ in range(300):
        p = [rng.uniform(-2, 3, 1), rng.uniform(-1, 2, 1), rng.uniform(-3, 3, 1)]
        g = [np.zeros(1) for _ in range(3)]
        e = bars_energy_grad(*p, bars, *g)
        if e == 0:
            continue
        for idx in range(3):
            q = [a.copy() for a in p]
            q[idx][0] += 1e-7
            e2 = bars_energy_grad(*q, bars, *[np.zeros(1) for _ in range(3)])
            num = (e2 - e) / 1e-7
            assert abs(num - g[idx][0]) < 1e-4 * max(1.0, abs(num))
        checked += 1
    assert checked > 20


def test_resting_grid_is_stable_under_gravity():
    n = 16
    i = np.arange(n)
    x, y, t = (i % 4 + 0.5), (i // 4 + 0.5), np.full(n, 0.01)
    v = [np.zeros(n) for _ in range(3)]
    simulate(x, y, t, *v, 4.0, 4.0, np.zeros((0, 4)), 2000, 1000.0, 0.0, -1.0, 1.0, 0.0, 0.004, 0.02, 1)
    *_, worst = energy_grad(x, y, t, 4.0, 0.0)
    assert worst < 0.01


def test_plate_leaves_holes():
    bars = _plate(6.0, 7.0, 1.25, 2.0, 0.0)
    lo = np.clip(bars[:, 0] - bars[:, 2], 0, 6)
    hi = np.clip(bars[:, 0] + bars[:, 2], 0, 6)
    assert abs((6.0 - (hi - lo).sum()) - 3 * 1.25) < 1e-9


def test_settle_and_close_give_feasible_packing():
    for mode in ("sieve", "cells", "shake"):
        x, y, t = settle(10, 3.8, np.random.default_rng(0), mode=mode, budget=6000)
        res = close_box(x, y, t)
        assert res is not None and res[3] <= 4.0 + 1e-6
