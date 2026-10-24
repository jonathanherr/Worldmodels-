"""Checks for the known constructions. Run with `python -m pytest tests`."""

import numpy as np
import pytest

from wm.constructions import R2, l_wrap, tilted_block
from wm.squares import official_check, polish, validate

# (side, n, tilted_block parameters): best-known packings from Friedman's survey.
KNOWN = [
    (3 + 1 / R2, 10, (1, 1, (-0.5, -0.5), "waves")),  # Goebel strip, a = 2
    ((7 + 3 * R2) / 2, 26, (3, 3, (-0.5, 0.0), "waves")),  # off-center 3 x 3 block
    (3 + 2 * R2, 28, (4, 4, (0.0, 0.0), "waves")),  # central 4 x 4 block, a = 2
    (6 + 1 / R2, 38, (1, 5, (-0.5, -0.5), "pairs")),  # Goebel strip, a = 5
    (5 + 5 / R2, 65, (5, 5, (0.0, 0.0), "waves")),  # central 5 x 5 block, a = 4
    (5.5 + 3 * R2, 85, (6, 6, (-0.5, 0.0), "waves")),  # off-center 6 x 6 block plus a column
    (5 + 7 / R2, 89, (7, 7, (0.0, 0.0), "waves")),  # central 7 x 7 block, a = 4
]


@pytest.mark.parametrize("side,n,cfg", KNOWN)
def test_reproduces_known_packing(side, n, cfg):
    x, y, deg = tilted_block(side, *cfg)
    assert len(x) == n
    r = validate(x, y, deg, tol=1e-8)
    assert r["valid"] and r["side"] <= side + 1e-8


def test_l_wrap_adds_an_l_and_passes_the_official_metric():
    side = 3 + 1 / R2
    x, y, deg = l_wrap(*tilted_block(side, 1, 1, (-0.5, -0.5), "waves"), side, layers=2)
    assert len(x) == 10 + 7 + 9  # 2 floor(s) + 1 per layer
    x, y, deg, report = polish(x, y, deg, gap=1e-9)
    assert report["valid"] and report["side"] < side + 2 + 1e-7
    rows = [{"id": k, "n": len(x), "square_id": k, "x": float(x[k]), "y": float(y[k]),
             "angle": float(np.deg2rad(deg[k]))} for k in range(len(x))]
    check = official_check(rows)
    assert check["valid"], check["reason"]
