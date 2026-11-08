import numpy as np

from wm.grow import GrowConfig, energy_grad, finish, grow
from wm.pack_opt import feasible


def test_gradient_matches_finite_differences():
    rng = np.random.default_rng(1)
    n = 8
    x, y = rng.uniform(0.1, 0.9, n), rng.uniform(0.1, 0.9, n)
    t, s = rng.uniform(0, 1.5, n), rng.uniform(0.25, 0.4, n)
    e, gx, gy, gt, _ = energy_grad(x, y, t, s, 1.0)
    assert e > 0
    h = 1e-7
    for arr, g in ((x, gx), (y, gy), (t, gt)):
        for i in range(n):
            arr[i] += h
            e2 = energy_grad(x, y, t, s, 1.0)[0]
            arr[i] -= h
            assert abs((e2 - e) / h - g[i]) < 1e-4


def test_growth_variants_jam_and_finish_feasible():
    for cfg in (GrowConfig(seed=0), GrowConfig(k=1, direction="side", adjacency="local", seed=1),
                GrowConfig(k=3, rate="contacts", adjacency="independent", direction="corner", seed=2)):
        r = grow(9, cfg)
        assert r["side_direct"] < 6.0
        x, y, t, L = finish(r)
        assert L < 4.5 and feasible(x, y, t, L)
