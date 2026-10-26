import numpy as np
import pytest

from wm import progsearch as ps
from wm.progsearch_lib import block, settle, worst_overlap


@pytest.mark.parametrize("corner", ["bl", "br", "tl", "tr"])
def test_local_frame_round_trip(corner):
    x, y, d = np.array([0.7, 3.2]), np.array([1.1, 9.0]), np.array([10.0, 45.0])
    lx, ly, ld = ps.to_local(x, y, d, 10.0, corner)
    gx, gy, gd = ps.to_global(lx, ly, ld, 10.0, corner)
    assert np.allclose(gx, x) and np.allclose(gy, y) and np.allclose(gd, d)


def test_block_and_settle_clear_overlaps():
    sq = block(0.0, 0.0, 2, 2) + [(0.9, 0.9, 30.0)]
    assert worst_overlap(sq) > 0.1
    out = settle(sq)
    assert len(out) == 5 and worst_overlap(out) < 1e-6
