import numpy as np

from wm import pack_opt, soft


def _random_layout(n=12, seed=1):
    rng = np.random.default_rng(seed)
    return (rng.uniform(0, 3, n), rng.uniform(0, 3, n), rng.uniform(-1, 1, n),
            rng.uniform(0.4, 0.5, n), rng.uniform(0.4, 0.5, n))


def test_gradient_matches_finite_differences():
    z = _random_layout()
    e, *g = soft.energy_grad(*z, 3.0, 3.0, 1e-9)
    assert e > 0
    h = 1e-6
    for v in range(5):
        for i in range(len(z[0])):
            zp = [q.copy() for q in z]
            zm = [q.copy() for q in z]
            zp[v][i] += h
            zm[v][i] -= h
            fd = (soft.energy_grad(*zp, 3.0, 3.0, 1e-9)[0] - soft.energy_grad(*zm, 3.0, 3.0, 1e-9)[0]) / (2 * h)
            assert abs(fd - g[v][i]) < 1e-6


def test_unit_squares_match_rigid_energy():
    x, y, t, _, _ = _random_layout()
    half = np.full(len(x), 0.5)
    e_soft = soft.energy_grad(x, y, t, half, half, 3.0, 1.0, 0.0)[0]
    e_rigid = pack_opt.energy_grad(x, y, t, 3.0, 0.0)[0]
    assert abs(e_soft - e_rigid) < 1e-9


def test_trial_recovers_loose_grid():
    g = np.arange(3) + 0.5
    X, Y = np.meshgrid(g, g)
    p = np.c_[X.ravel() * 1.05, Y.ravel() * 1.05, np.zeros(9)]
    r, deficit = soft.soft_trial(p, 3.15, 0.04, (1.0, 100.0, 1e4), 100, 1e-3, True, 0)
    assert r is not None
    assert r[3] < 3.0 + 1e-3
    assert deficit.shape == (9,)
