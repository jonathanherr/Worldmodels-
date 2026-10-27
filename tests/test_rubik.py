import numpy as np

from wm import rubik
from wm.pack_opt import feasible


def grid(k):
    return np.array([(i + 0.5, j + 0.5, 0.0) for i in range(k) for j in range(k)])


def test_dihedral_keeps_grid_feasible():
    p = grid(3)
    for k in range(8):
        q = rubik.dihedral(p, 3.0, k)
        assert feasible(q[:, 0], q[:, 1], q[:, 2], 3.0 + 1e-9, margin=0.0)


def test_clean_block_turn_is_exact():
    p = grid(4)
    q, m = rubik.block_turn(p, 0.0, 0.0, 2.0, 1)
    assert m == 4
    assert feasible(q[:, 0], q[:, 1], q[:, 2], 4.0 + 1e-9, margin=0.0)
    assert sorted(map(tuple, np.round(q[:, :2], 9))) == sorted(map(tuple, np.round(p[:, :2], 9)))


def test_slice_turn_swaps_band_and_can_change_count():
    p = grid(3)
    other = np.r_[grid(3)[grid(3)[:, 1] > 1], [[1.5, 0.5, 0.0]]]  # bottom row holds one square
    q = rubik.slice_turn(p, other, 1, -1.0, 1.0)
    assert len(q) == 7


def test_straddle_profile_finds_clean_cuts():
    cs, cnt = rubik.straddle_profile(grid(3), 0, 3.0, res=0.5)
    assert cnt[cs == 1.0][0] == 0 and cnt[cs == 1.5][0] == 3
