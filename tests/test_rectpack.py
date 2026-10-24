"""Checks for the rectangle-decomposition packer. Run with `python -m pytest tests`."""

import numpy as np
import pytest

from wm.rectpack import Layout, compose, lean_angle, rect_pack, wrap_core
from wm.squares import official_check, validate


def _official(x, y, deg):
    n = len(x)
    rows = [{"id": k, "n": n, "square_id": k, "x": float(x[k]), "y": float(y[k]),
             "angle": float(np.deg2rad(deg[k]))} for k in range(n)]
    return official_check(rows)


def test_lean_angle_fits_the_stack():
    a = lean_angle(4.9, 5)
    assert 5 * np.cos(a) + np.sin(a) <= 4.9


@pytest.mark.parametrize("w,L", [(2.914, 12.0), (4.95, 26.0), (3.9, 25.5)])
def test_rect_pack_beats_axis_and_stays_inside(w, L):
    x, y, d, _ = rect_pack(w, L)
    assert len(x) > int(w) * int(L)
    r = validate(x, y, d, tol=0.0)
    assert r["valid"]
    from wm.rectpack import square_corners
    c = square_corners(x, y, d)
    assert c[..., 0].min() >= 0 and c[..., 0].max() <= L and c[..., 1].min() >= 0 and c[..., 1].max() <= w


def test_compose_passes_the_official_metric():
    x, y, d = compose(Layout(9.9, 4, 3))
    r = _official(x, y, d)
    assert r["valid"], r["reason"]
    assert r["side"] <= 9.9 + 1e-9


def test_wrap_core_passes_the_official_metric():
    g = np.arange(6) + 0.5
    gx, gy = np.meshgrid(g, g, indexing="ij")
    x, y, d = wrap_core((gx.ravel(), gy.ravel(), np.zeros(36)), 6.0, 3.9)
    r = _official(x, y, d)
    assert r["valid"], r["reason"]
    assert len(x) > 36 + 3 * 6 + 3 * 9  # beats an axis-aligned L of width 3
