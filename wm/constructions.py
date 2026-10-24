"""Known constructions for packing unit squares in a square, for any n.

Every construction here is one family: inside a container [0, s]^2 sits a block of squares
tilted 45 degrees (p rows across the main diagonal, q squares along it), and the rest of the
container is filled with axis-aligned unit cells on four lattices, each anchored at one corner.
The tilted rows are then extended along the diagonal wherever there is still room. This one
family covers the general constructions in Friedman's survey "Packing Unit Squares in Squares":

* Goebel's diagonal strip (p = 1, full length): s = a + 1 + 1/sqrt2 holds
  a^2 + a + 3 + floor((a - 1) sqrt2) squares (n = 10, 38, 52, 67, 84, ...).
* Goebel's central b x b tilted block (p = q = b): s = a + 1 + b/sqrt2 holds 2a^2 + 2a + b^2
  (n = 28, 40, 65, 89, ...).
* Wider diagonal strips (p = 2..4), as used for n = 66 and 87.
* Off-center blocks (`shift`), which is how a block packing plus an extra column comes out.

On top of that, `l_wrap` adds an L of axis-aligned squares around any packing (n + 2 floor(s) + 1
squares in side s + 1), and any packing of N squares also packs every n < N (drop squares).

Squares are allowed to touch here; `wm.squares.polish` opens the 1e-9 gap that Kaggle's
interval-arithmetic metric needs before anything is written out.
"""

import numpy as np

R2 = np.sqrt(2.0)
EPS = 1e-9  # overlap tolerance while constructing; polish removes anything this small

# Two unit squares, one axis-aligned and one at 45 degrees, overlap unless some separating
# axis splits them: |dx| or |dy| >= 1/2 + 1/sqrt2, or |dx +- dy| >= 1 + 1/sqrt2.
_AX_TILT = 0.5 + 1 / R2
_DIAG_TILT = 1 + 1 / R2


def _axis_vs_tilted(ax, tc):
    """(A,) bool: does each axis-aligned square (centers ax, (A, 2)) hit any 45-degree square in tc?"""
    if len(ax) == 0 or len(tc) == 0:
        return np.zeros(len(ax), bool)
    d = ax[:, None, :] - tc[None, :, :]
    dx, dy = d[..., 0], d[..., 1]
    sep = ((np.abs(dx) >= _AX_TILT - EPS) | (np.abs(dy) >= _AX_TILT - EPS)
           | (np.abs(dx + dy) >= _DIAG_TILT - EPS) | (np.abs(dx - dy) >= _DIAG_TILT - EPS))
    return ~sep.all(1)


def _axis_vs_axis(a, b):
    """(A,) bool: does each axis-aligned square in a hit any axis-aligned square in b?"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros(len(a), bool)
    d = np.abs(a[:, None, :] - b[None, :, :])
    return ((d[..., 0] < 1 - EPS) & (d[..., 1] < 1 - EPS)).any(1)


def _inside_tilted(tc, s):
    h = 1 / R2  # half-extent of a 45-degree unit square along x and y
    return ((tc >= h - EPS) & (tc <= s - h + EPS)).all(1)


def tilted_block(s: float, p: int, q: int, shift=(0.0, 0.0), order: str = "waves", extend: bool = True):
    """Centers (x, y) and angles (degrees) of the tilted-block packing in a container of side s.

    p rows of 45-degree squares run along the main diagonal, q squares per row, centered at the
    container's middle plus `shift`. Corner lattices fill the rest; with `extend`, the tilted rows
    then grow along the diagonal into any space the lattices left. `order` decides how the four
    lattices share the space: "waves" grows all four corners together, "pairs" grows the
    top-left and bottom-right staircases first and gives the other two corners what is left.
    """
    u = np.array([1.0, 1.0]) / R2  # along the diagonal
    v = np.array([-1.0, 1.0]) / R2  # across it
    c0 = np.array([s / 2, s / 2]) + np.asarray(shift, float)
    rows = np.arange(p) - (p - 1) / 2

    def row_centers(js):
        i, j = np.meshgrid(rows, js, indexing="ij")
        return (c0 + j[..., None] * u + i[..., None] * v).reshape(-1, 2), i.ravel()

    tc, ti = row_centers(np.arange(q) - (q - 1) / 2) if q > 0 and p > 0 else (np.zeros((0, 2)), np.zeros(0))
    keep = _inside_tilted(tc, s)
    tc, ti = tc[keep], ti[keep]

    # Axis-aligned lattices anchored at each corner, filled together in waves of growing distance
    # from their own corner, so each corner grows a staircase until it meets its neighbors.
    m = int(np.floor(s + EPS))
    k = np.arange(m)
    gi, gj = np.meshgrid(k, k, indexing="ij")
    gi, gj = gi.ravel(), gj.ravel()
    lattices = []
    for fx, fy in [(0, 1), (1, 0), (0, 0), (1, 1)]:  # TL, BR, BL, TR
        cx = s - (gi + 0.5) if fx else gi + 0.5
        cy = s - (gj + 0.5) if fy else gj + 0.5
        cells = np.stack([cx, cy], 1)
        ok = ~_axis_vs_tilted(cells, tc)
        lattices.append((cells[ok], (gi + gj)[ok]))
    axis = np.zeros((0, 2))
    groups = [lattices] if order == "waves" else [lattices[:2], lattices[2:]]
    for group in groups:
        for wave in range(2 * m):
            for cells, dist in group:
                c = cells[dist == wave]
                if len(c) == 0:
                    continue
                c = c[~_axis_vs_axis(c, axis)]
                axis = np.concatenate([axis, c])

    if extend and p > 0:
        reach = int(np.ceil(s * R2)) + 2
        cand, ci = row_centers(np.arange(-reach, reach + 1) + (0.5 if q % 2 == 0 else 0.0))
        cand = cand[_inside_tilted(cand, s)]
        # A candidate must not overlap a tilted square already placed (same-row spacing 1 along u,
        # rows spaced 1 apart along v, so on this lattice that means "not the same spot").
        if len(tc):
            dup = (np.abs(cand[:, None, :] - tc[None, :, :]).sum(-1) < 1e-6).any(1)
            cand = cand[~dup]
        hit = np.zeros(len(cand), bool)
        if len(axis):
            d = cand[:, None, :] - axis[None, :, :]
            dx, dy = d[..., 0], d[..., 1]
            sep = ((np.abs(dx) >= _AX_TILT - EPS) | (np.abs(dy) >= _AX_TILT - EPS)
                   | (np.abs(dx + dy) >= _DIAG_TILT - EPS) | (np.abs(dx - dy) >= _DIAG_TILT - EPS))
            hit = ~sep.all(1)
        tc = np.concatenate([tc, cand[~hit]])

    x = np.concatenate([axis[:, 0], tc[:, 0]])
    y = np.concatenate([axis[:, 1], tc[:, 1]])
    deg = np.concatenate([np.zeros(len(axis)), np.full(len(tc), 45.0)])
    return x, y, deg


def l_wrap(x, y, deg, side: float, layers: int = 1):
    """Wrap `layers` L-shaped layers of axis-aligned squares around a packing of side `side`.

    The packing must sit in [0, side]^2. Each layer adds 2 floor(side) + 1 squares and 1 to the side.
    """
    x, y, deg = (list(np.asarray(a, float)) for a in (x, y, deg))
    s = side
    for _ in range(layers):
        m = int(np.floor(s + EPS))
        for r in range(m + 1):  # right column, full new height s + 1
            x.append(s + 0.5)
            y.append(s + 1 - 0.5 - r)
            deg.append(0.0)
        for c in range(m):  # top row over the old packing
            x.append(c + 0.5)
            y.append(s + 0.5)
            deg.append(0.0)
        s += 1
    return np.array(x), np.array(y), np.array(deg)


# Shifts of the tilted block that matter, up to the container's symmetries (reflection in the
# main diagonal swaps dx and dy, a half turn negates both).
SHIFTS = [(0.0, 0.0), (-0.5, 0.0), (-0.5, -0.5), (-0.5, 0.5), (-1.0, 0.0), (-1.0, -0.5), (-1.0, 0.5),
          (-1.0, -1.0), (-1.0, 1.0)]


def sqrt2_part(s: float) -> int:
    """j such that s = (i + j sqrt2) / 2 for an integer i (s must come from candidate_sides)."""
    for j in range(0, 200):
        i = 2 * s - j * R2
        if abs(i - round(i)) < 1e-6:
            return j
    raise ValueError(f"{s} is not of the form (i + j sqrt2)/2")


def configs(s: float):
    """Parameters (p, q, shift, order) of `tilted_block` worth trying at side s.

    In every construction found by a full sweep for s < 12, the side is (i + j sqrt2)/2 with
    p within one of j, strips (p <= 4) run nearly the full diagonal or form a p x p block, and
    wider blocks are (nearly) square. That prunes the sweep from tens of thousands of
    configurations per side to a few hundred.
    """
    j = sqrt2_part(s)
    full = int(s * R2) + 1
    for p in range(max(1, j - 1), j + 2):
        qs = {p - 1, p, p + 1}
        if p <= 4:
            qs |= set(range(full - 6, full + 4))
        for q in sorted(q for q in qs if 0 <= q <= full + 4):
            for shift in SHIFTS:
                for order in ("waves", "pairs"):
                    yield p, q, shift, order


def useful_j(s: float, j: int) -> bool:
    """Strips (j <= 5) and blocks near Goebel's sweet spot b ~ s / sqrt2 are the only sides
    whose best count beats the grid in the full sweep."""
    return j <= 5 or 0.707 * (s - 2) - 2 <= j <= 0.707 * s + 2


def candidate_sides(s_max: float, frac_min: float = 0.0):
    """Side lengths of the form (i + j sqrt2) / 2 below s_max: where constraints in these
    45-degree constructions go tight. Only sides with fractional part >= frac_min are kept
    (integer sides never beat the grid)."""
    out = set()
    for j in range(0, int(2 * s_max / R2) + 2):
        for i in range(-2 * int(s_max) - 2, 2 * int(s_max) + 3):
            s = (i + j * R2) / 2
            if 1 <= s < s_max and s - np.floor(s) >= frac_min and useful_j(s, j):
                out.add(s)
    return sorted(out)
