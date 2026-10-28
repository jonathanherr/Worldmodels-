import numpy as np

from wm.blockpack import Blocks, BlockPacker
from wm.squares import official_check


def _grid_packs(max_n):
    packs = {}
    for n in range(1, max_n + 1):
        g = int(np.ceil(np.sqrt(n)))
        i = np.arange(n)
        e = 1 + 1e-7  # spaced a hair over 1: offset blocks must not touch exactly
        packs[n] = ((i % g) * e + 0.5, (i // g) * e + 0.5, np.zeros(n), g * e)
    # the 5-square record (side 2 + 1/sqrt2) as one tilted block
    r = 0.5 * np.sqrt(2)
    s5 = 2 + r + 1e-6  # hair over the exact record so the tilted square never touches
    packs[5] = (np.array([0.5, s5 - 0.5, 0.5, s5 - 0.5, s5 / 2]), np.array([0.5, 0.5, s5 - 0.5, s5 - 0.5, s5 / 2]),
                np.array([0, 0, 0, 0, np.pi / 4]), s5)
    return packs


def test_blocks_compose_into_valid_packing():
    packs = _grid_packs(40)
    raw = [0.0] + [packs[n][3] for n in range(1, 41)]
    bp = BlockPacker(Blocks(raw, lambda k: packs[k]))
    # a column plus the 3x3 block: the plan must pass the official metric with exactly n squares
    n = 12
    L = bp.min_side(n, 4 + 1e-6, tol=1e-7)
    assert L is not None and L <= 4 + 1e-6
    x, y, t = bp.build(L + 1e-7, n)
    rows = [{"id": k, "n": n, "square_id": k, "x": float(x[k]), "y": float(y[k]), "angle": float(t[k])}
            for k in range(n)]
    check = official_check(rows)
    assert check["valid"], check
    assert check["side"] <= L + 1e-6


def test_record_block_is_reused_for_its_own_n():
    packs = _grid_packs(10)
    raw = [0.0] + [packs[n][3] for n in range(1, 11)]
    bp = BlockPacker(Blocks(raw, lambda k: packs[k]))
    L = bp.min_side(5, 3 + 1e-6, tol=1e-8)
    assert abs(L - (2 + 0.5 * np.sqrt(2))) < 1e-5
