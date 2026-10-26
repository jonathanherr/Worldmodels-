import numpy as np

from wm import fault_dp as fd
from wm.squares import five_solution


def _five():
    x, y, deg = five_solution()
    return fd.normalize(np.stack([x, y, np.deg2rad(deg)], 1))


def test_symmetries_keep_the_side():
    p = _five()
    for q in fd.symmetries(p):
        assert abs(fd.side_of(q) - fd.side_of(p)) < 1e-12


def test_five_gadget_chains_into_valid_packings():
    donors = [_five()]
    gadgets = fd.extract(0, donors[0], max_wh=4, max_d=4)
    g = min((g for g in gadgets if (g.w, g.h, g.d) == (2, 2, 1)), key=lambda g: g.f)
    assert abs(g.f - (1 / np.sqrt(2))) < 1e-9
    groups = fd.build_groups(gadgets, max_d=4)
    layers = fd.run_dp(groups, 1, max_wh=4, max_d=4)
    # one gadget: 3x3 grid + 1 = 10 squares at side 3 + 1/sqrt(2) (the known optimum)
    # two gadgets need a buffer column (no free tips): 5x5 grid + 2 = 27 squares at 5 + 1/sqrt(2)
    for K, D in [(3, 1), (5, 2)]:
        f, r, b, W, H = fd.best_chains(layers, K, D)
        assert abs(f - 1 / np.sqrt(2)) < 1e-9
        chain = fd.backtrack(layers, groups, r, b, W, H, D, f)
        sq, f2 = fd.compose(chain, donors, K)
        n = K * K + D
        assert len(sq) == n
        side, reason = fd.check_rows(n, fd.to_rows(n, sq))
        assert reason == "ok", reason
        assert abs(side - (K + 1 / np.sqrt(2))) < 1e-9
