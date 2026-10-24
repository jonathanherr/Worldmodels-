"""Checks for the dimensional-relaxation (lifted) energy and the lift-and-flatten pipeline."""

import numpy as np
import pytest

from wm.lift import energy_grad_lift, lift_compress
from wm.pack_opt import energy_grad, feasible
from wm.squares import grid_solution


@pytest.mark.parametrize("lens", [True, False])
def test_lifted_gradient_matches_finite_differences(lens):
    rng = np.random.default_rng(0)
    n, L, H, h, kappa = 10, 3.0, 1.0, 1.0, 0.3
    v = np.concatenate([rng.uniform(0, L, n), rng.uniform(0, L, n), rng.uniform(-0.7, 0.7, n),
                        rng.uniform(-0.2, 1.2, n), [L]])

    def energy(v):
        return energy_grad_lift(v[:n], v[n:2 * n], v[2 * n:3 * n], v[3 * n:4 * n], v[-1], H, h, kappa, lens, 0.0)

    _, gx, gy, gt, gz, gL, _ = energy(v)
    g = np.concatenate([gx, gy, gt, gz, [gL]])
    num = np.zeros_like(v)
    for k in range(len(v)):
        vp, vm = v.copy(), v.copy()
        vp[k] += 1e-6
        vm[k] -= 1e-6
        num[k] = (energy(vp)[0] - energy(vm)[0]) / 2e-6
    assert np.abs(num - g).max() < 1e-6


def test_flat_heights_give_the_2d_energy():
    rng = np.random.default_rng(1)
    n, L = 9, 2.6
    x, y, t = rng.uniform(0, L, n), rng.uniform(0, L, n), rng.uniform(-1, 1, n)
    e2d = energy_grad(x, y, t, L, 0.0)[0]
    for lens in (True, False):
        e3d = energy_grad_lift(x, y, t, np.zeros(n), L, 0.0, 1.0, 0.5, lens, 0.0)[0]
        assert np.isclose(e2d, e3d)


def test_slabs_one_apart_do_not_interact():
    x, y, t = np.array([1.0, 1.0]), np.array([1.0, 1.0]), np.zeros(2)
    e = energy_grad_lift(x, y, t, np.array([0.0, 1.0]), 2.0, 1.0, 1.0, 0.0, False, 0.0)[0]
    assert e == 0.0


@pytest.mark.parametrize("H0", [0.0, 1.0])
def test_lift_compress_fits_a_loose_grid(H0):
    x, y, deg = grid_solution(9)  # 3x3 grid, side 3
    x, y = x * 1.2, y * 1.2  # loosen to side 3.6, then ask for 3.05
    r = lift_compress(x, y, np.deg2rad(deg), 3.6, 3.05, H0=H0, rng=np.random.default_rng(0))
    assert r is not None and feasible(*r, 3.05)
