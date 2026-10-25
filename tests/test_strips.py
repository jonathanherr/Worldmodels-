import numpy as np

from wm import strips
from wm.squares import official_check


def _rows(x, y, d):
    return [{"id": k, "n": len(x), "square_id": k, "x": float(x[k]), "y": float(y[k]),
             "angle": float(np.deg2rad(d[k]))} for k in range(len(x))]


def test_band_never_below_grid():
    for W, H in [(5.5, 2.3), (12.0, 3.9), (31.8, 15.8)]:
        n, _, _ = strips.best_band(W, H, phases=8)
        assert n >= int(W) * int(H)


def test_tall_band_beats_grid():
    # Columns of 16 squares tilted ~13 degrees fit a 15.8-high band: more than 31 * 15 = 465.
    n, theta, _ = strips.best_band(31.8, 15.8, phases=24)
    assert n > 465 and theta > 0


def test_vectorised_count_matches_columns():
    n, theta, t0 = strips.best_band(20.7, 9.8, phases=12)
    assert theta > 0
    assert strips.band_count(20.7, 9.8, theta, t0) == n


def test_layout_is_official_valid_and_counts_match():
    S, A, B = 12.95, 9, 9
    n, top, right = strips.layout_count(S, A, B, phases=12)
    x, y, d = strips.layout(S, A, B, top, right)
    assert len(x) == n
    check = official_check(_rows(x, y, d))
    assert check["valid"], check["reason"]
    assert check["side"] <= S + 1e-9
