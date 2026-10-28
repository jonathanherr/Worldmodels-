"""Checks for the local optimizer, the neuroevolution decoder and the best-per-n store."""

import numpy as np

from wm import best_store
from wm.neuroevo import NetShape, decode, features
from wm.pack_opt import energy_grad, feasible, squeeze
from wm.squares import five_solution, grid_solution, polish, validate


def test_energy_gradient_matches_finite_differences():
    rng = np.random.default_rng(0)
    n, L = 8, 2.5
    z = np.concatenate([rng.uniform(0, L, n), rng.uniform(0, L, n), rng.uniform(-1, 1, n)])
    _, gx, gy, gt, _, _ = energy_grad(z[:n], z[n:2 * n], z[2 * n:], L, 0.0)
    g = np.concatenate([gx, gy, gt])
    num = np.zeros_like(z)
    for k in range(len(z)):
        zp, zm = z.copy(), z.copy()
        zp[k] += 1e-6
        zm[k] -= 1e-6
        num[k] = (energy_grad(zp[:n], zp[n:2 * n], zp[2 * n:], L, 0.0)[0]
                  - energy_grad(zm[:n], zm[n:2 * n], zm[2 * n:], L, 0.0)[0]) / 2e-6
    assert np.abs(num - g).max() < 1e-6


def test_energy_depth_agrees_with_checker():
    x, y, deg = [0.0, 1.1], [0.0, 0.0], [0.0, 45.0]
    *_, worst = energy_grad(np.array(x) + 5, np.array(y) + 5, np.deg2rad(deg), 20.0, 0.0)
    assert np.isclose(worst, 0.5 + np.sqrt(2) / 2 - 1.1)


def test_squeeze_turns_a_loose_layout_into_a_valid_packing():
    x, y, deg = grid_solution(4)
    r = squeeze(x * 1.3, y * 1.3, np.deg2rad(deg), 2.6)
    assert r is not None
    x, y, t, L = r
    assert feasible(x, y, t, L)
    assert L < 2.0 + 1e-5
    px, py, pd, rep = polish(x, y, np.rad2deg(t))
    assert rep["valid"] and rep["side"] < 2.0 + 1e-5


def test_zero_genome_decodes_to_the_grid():
    shape = NetShape(hidden=4)
    x, y, t, L0 = decode(np.zeros(shape.size), 7, shape)
    assert features(7).shape == (7, shape.n_in)
    assert np.allclose(t, 0)
    cell = L0 / 3
    assert np.allclose(np.sort(np.unique(np.round(x / cell - 0.5, 6))), [0, 1, 2])


def test_store_keeps_only_certified_improvements(tmp_path):
    assert best_store.best_side(5, tmp_path) == 3.0
    x, y, d, _ = polish(*five_solution())
    assert best_store.offer(5, x, y, d, "test", root=tmp_path)["accepted"]
    assert not best_store.offer(5, x, y, d, "test", root=tmp_path)["accepted"]  # not better
    assert not best_store.offer(6, *grid_solution(6), "test", root=tmp_path)["accepted"]  # equals grid
    bad = five_solution()
    bad[2][4] = 40.0  # overlapping
    assert not best_store.offer(5, *bad, "test", root=tmp_path / "other")["accepted"]
    xs, ys, ds = best_store.load_xyd(5, tmp_path)
    assert validate(xs, ys, ds)["valid"]
    rows = best_store.submission_rows(tmp_path, max_n=6)
    assert len(rows) == 21 and rows[10]["x"] == best_store.load(5, tmp_path)["x"][0]


def test_neat_operators_keep_networks_acyclic_and_empty_genome_is_the_grid():
    from wm import neat

    rng = np.random.default_rng(0)
    inn = neat.Innovations()
    cfg = neat.NeatConfig(p_add_conn=0.9, p_add_node=0.6, p_act=0.5, p_toggle=0.3)
    pop = [neat.initial_genome(rng, inn, cfg) for _ in range(8)]
    feats = features(9)
    for _ in range(30):
        a, b = rng.choice(len(pop), 2, replace=False)
        child = neat.mutate(neat.crossover(pop[a], pop[b], rng), rng, inn, cfg)
        out = neat.activate(child, feats)  # raises on a cycle
        assert out.shape == (9, 3) and np.all(np.abs(out) <= 1)
        pop[int(rng.integers(len(pop)))] = child
    assert any(k not in neat.OUTPUTS for g in pop for k in g.nodes)  # some hidden nodes were added
    empty = neat.Genome({k: ["identity", 0.0] for k in neat.OUTPUTS})
    x, y, t, L0 = neat.decode(empty, 9)
    assert np.allclose(t, 0) and np.allclose(np.sort(np.unique(np.round(x / (L0 / 3) - 0.5, 6))), [0, 1, 2])
