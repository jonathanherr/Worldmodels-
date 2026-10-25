import numpy as np

from wm import morph
from wm.pack_opt import energy_grad, feasible
from wm.squares import official_check


def test_sized_energy_matches_unit_energy():
    rng = np.random.default_rng(1)
    n = 20
    x, y, t = rng.uniform(0, 4, n), rng.uniform(0, 4, n), rng.uniform(-1, 1, n)
    a = morph.energy_grad_sized(x, y, t, np.ones(n), 4.0, 2e-9)
    b = energy_grad(x, y, t, 4.0, 2e-9)
    for k in range(5):
        assert np.allclose(a[k], b[k])


def test_sized_gradient_matches_finite_differences():
    rng = np.random.default_rng(2)
    n = 12
    x, y, t = rng.uniform(0, 3, n), rng.uniform(0, 3, n), rng.uniform(-1, 1, n)
    s = rng.uniform(0.3, 1.0, n)
    e, gx, gy, gt, _ = morph.energy_grad_sized(x, y, t, s, 3.0, 0.0)
    z, g = np.concatenate([x, y, t]), np.concatenate([gx, gy, gt])
    num = []
    for k in range(3 * n):
        zz = z.copy()
        zz[k] += 1e-7
        num.append((morph.energy_grad_sized(zz[:n], zz[n:2 * n], zz[2 * n:], s, 3.0, 0.0)[0] - e) / 1e-7)
    assert np.abs(np.array(num) - g).max() < 1e-4


def test_absent_square_costs_nothing():
    x, y, t = np.array([0.5, 0.5]), np.array([0.5, 0.6]), np.zeros(2)
    e, *_ = morph.energy_grad_sized(x, y, t, np.array([1.0, 0.0]), 1.0, 0.0)
    assert e == 0.0


def test_remove_fractional_returns_feasible_packing():
    k = 3
    i = np.arange(k * k)
    x, y, t = (i % k + 0.5).astype(float), (i // k + 0.5).astype(float), np.zeros(k * k)
    r = morph.remove_fractional(x, y, t, float(k), 4, steps=3)
    assert r is not None
    rx, ry, rt, L = r
    assert len(rx) == 8 and feasible(rx, ry, rt, L)


def test_compose_2x2_blocks_are_valid():
    # Block b: 4 squares loosely spread in a 2.5 box, so the leftover rectangles have slack.
    sides = np.array([0.0, 1.0, 2.0, 2.0, 2.5, 3.0])

    def packings(m):
        if m == 4:
            return np.array([0.6, 1.9, 0.6, 1.9]), np.array([0.6, 0.6, 1.9, 1.9]), np.zeros(4), 2.5
        return np.array([0.5]), np.array([0.5]), np.zeros(1), 1.0

    x, y, t, S = morph.compose_2x2(1, 4, sides, packings)
    assert len(x) == 1 + 4 + 2 * morph.rect_count(1.0, 2.5, sides) == 9
    rows = [{"id": k, "n": 9, "square_id": k, "x": float(x[k]), "y": float(y[k]), "angle": 0.0} for k in range(9)]
    check = official_check(rows)
    assert check["valid"], check["reason"]
    assert check["side"] <= S + 1e-9


def test_no_composition_beats_the_grid_with_grid_blocks():
    sides = np.array([0.0] + [float(np.ceil(np.sqrt(n))) for n in range(1, 201)])
    assert morph.composition_wins(sides, 200) == {}
