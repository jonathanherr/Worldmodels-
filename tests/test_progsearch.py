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


def test_construct_tasks_hide_the_record():
    for task in ("construct-strip", "construct335", "construct147"):
        assert ps.TASKS[task].get("hide")
    ctx = ps.make_ctx(147, "bl", None, select="all", hide=True)
    assert ctx["original"] == [] and ctx["k"] == 147


def test_gadget_bloat_and_pool_merge():
    import math

    from wm import fault_dp as fd
    from wm import gadget_search as gs

    s = 1.5 + math.sqrt(2) / 2
    five = np.array([[0.5, 0.5, 0], [s, 0.5, 0], [0.5, s, 0], [s, s, 0], [(s + 0.5) / 2, (s + 0.5) / 2, math.pi / 4]])
    assert abs(gs.gadget_f(five, 2, 2) - math.sqrt(2) / 2) < 1e-12
    worse = fd.Gadget(2, 2, 1, 0.9, True, True, 0, 0)
    better = fd.Gadget(2, 2, 1, 0.8, True, True, 1, 0)
    pool = {"groups": fd.build_groups([worse])}
    merged = gs._merge(pool["groups"], [better])
    assert merged[(2, 2, True, True)][0][1] == 0.8 and pool["groups"][(2, 2, True, True)][0][1] == 0.9
    assert gs.best_known({"groups": merged}, 2, 2, 1) == 0.8
