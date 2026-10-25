"""Checks for the Game-of-Life packing rules: locality, determinism, feasible finishes, genome round trip."""

import numpy as np

from wm import life
from wm import life_evolve as le
from wm.pack_opt import feasible


def test_update_is_local():
    """Moving a far-away square does not change what a square sees."""
    rng = np.random.default_rng(0)
    x, y, t, L = life.random_seed(30, rng, loose=1.1)
    c = life.decode(life.from_dict({"coh": 0.05, "align": 0.05, "roll": 0.5, "slide": 0.02, "space": 0.05}))
    kicks = np.zeros((3, 30))
    fx, fy, ft, _ = life.local_update(x, y, t, L, c, 0.0, kicks)
    d = np.hypot(x - x[0], y - y[0])
    far = int(np.argmax(d))
    assert d[far] > 4
    x2 = x.copy()
    x2[far] += 0.3
    gx, gy, gt, _ = life.local_update(x2, y, t, L, c, 0.0, kicks)
    near0 = d < 4 - 0.3 - 2  # squares whose neighbourhood cannot reach the moved one
    near0[far] = False
    assert near0.sum() > 0
    assert np.allclose(fx[near0], gx[near0]) and np.allclose(ft[near0], gt[near0])


def test_rule_is_deterministic_and_finishes_feasible():
    rng = np.random.default_rng(1)
    x, y, t, L = life.random_seed(12, rng)
    c = le.library_rule("granular", path="/nonexistent.json")
    a = life.run_rule(c, x, y, t, L, steps=200, seed=3)
    b = life.run_rule(c, x, y, t, L, steps=200, seed=3)
    assert all(np.array_equal(u, v) for u, v in zip(a[:3], b[:3]))
    r = le.finish(*a)
    assert r is not None and feasible(*r) and r[3] < L


def test_genome_round_trip():
    for inv in le.INVENTORS.values():
        c = life.decode(life.from_dict(inv.start))
        back = le.rule_from(inv, le.genome_from(inv, c))
        free = [life.NAMES.index(k) for k in inv.free]
        assert np.allclose(back[free], c[free], atol=1e-3 * (life.HIGH - life.LOW)[free].max())
