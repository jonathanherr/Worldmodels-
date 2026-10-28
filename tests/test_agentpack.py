import numpy as np

from wm.agentpack import agent_pack, coalitions, stress
from wm.pack_opt import feasible


def test_agent_pack_beats_grid_at_n10():
    x, y, t, L, stats = agent_pack(10, seed=0)
    assert feasible(x, y, t, L)
    assert L < 3.9  # grid is 4, record 3.7071
    assert stats["tries"] > 0


def test_stress_and_coalitions_on_a_grid():
    g = np.arange(3) + 0.5
    x, y = (a.ravel() for a in np.meshgrid(g, g))
    t = np.zeros(9)
    assert stress(x, y, t, 3.0, 0.0).max() == 0.0
    assert len(coalitions(x, y, t)[0]) == 9
    t[4] = np.pi / 4  # a tilted centre square overlaps all four neighbours and leaves the group
    assert stress(x, y, t, 3.0, 0.0)[4] > 0
    assert 4 not in coalitions(x, y, t)[0]
