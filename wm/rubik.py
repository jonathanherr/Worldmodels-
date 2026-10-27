"""Rubik's-cube moves for square packings: turn slices and blocks, then let the packing settle.

Picture the packing of n squares in [0, L]^2 as the front face of a cube whose other faces hold
copies of packings in other orientations (the 8 dihedral images of the same packing, or a
packing for a neighbouring n with a similar side). A cube move then has a 2D meaning:

- slice turn: a band of rows (or columns) of the front face is replaced by the same band of
  another face. Turning a slice 180 degrees brings in the mirrored band; a quarter turn brings
  in the band of a rotated copy or of another packing. The square count may change (n -> n'),
  so a move made on n can land on a neighbouring puzzle.
- block turn: a square window of the face is rotated by 90, 180 or 270 degrees or reflected
  about its centre, like turning one face of a small cube embedded in the big one.

"Projecting down" is the settle step: the turned layout overlaps along the cut lines, so
`pack_opt.compact_squeeze` (box side free, penalty relax, then an exactly feasible squeeze)
flattens it back to a valid packing. A rigid move keeps the side only when its cuts are clean;
the point of a move is to regroup slack (holes in one band line up with holes in another) so
the settle step can shrink the box below where perturbation polish snaps back to.
"""

import numpy as np

from .pack_opt import compact_squeeze

HALF_PI = np.pi / 2


def half_ext(t):
    return (np.abs(np.cos(t)) + np.abs(np.sin(t))) / 2


def side_of(p):
    h = half_ext(p[:, 2])
    return float(max((p[:, 0] + h).max() - (p[:, 0] - h).min(), (p[:, 1] + h).max() - (p[:, 1] - h).min()))


def normalize(p):
    p = np.array(p, dtype=float)
    h = half_ext(p[:, 2])
    p[:, 0] -= (p[:, 0] - h).min()
    p[:, 1] -= (p[:, 1] - h).min()
    return p


def dihedral(p, L, k):
    """k in 0..7: rotate by k%4 quarter turns about the box centre, then mirror x if k >= 4."""
    q = np.array(p, dtype=float)
    for _ in range(k % 4):
        q = np.c_[L - q[:, 1], q[:, 0], q[:, 2] + HALF_PI]
    if k >= 4:
        q = np.c_[L - q[:, 0], q[:, 1], -q[:, 2]]
    return q


def straddle_profile(p, axis, L, res=0.02):
    """Positions c (step res) and how many squares cross the line {coord_axis = c}."""
    c = np.arange(0.0, L + res, res)
    h = half_ext(p[:, 2])
    lo, hi = p[:, axis] - h, p[:, axis] + h
    cnt = ((lo[None, :] < c[:, None] - 1e-9) & (hi[None, :] > c[:, None] + 1e-9)).sum(1)
    return c, cnt


def pick_cut(cs, cnt, rng, lo, hi, tau=1.0):
    """A cut position in [lo, hi], favouring lines few squares cross (Boltzmann weight)."""
    m = (cs >= lo) & (cs <= hi)
    if not m.any():
        return None
    w = np.exp(-(cnt[m] - cnt[m].min()) / tau)
    return float(rng.choice(cs[m], p=w / w.sum()))


def slice_turn(front, other, axis, a, b):
    """Front with the band a <= centre_axis < b replaced by the same band of `other`."""
    keep = ~((front[:, axis] >= a) & (front[:, axis] < b))
    take = (other[:, axis] >= a) & (other[:, axis] < b)
    return np.r_[front[keep], other[take]]


def block_turn(p, x0, y0, s, k):
    """Rotate/reflect (dihedral k, as in `dihedral`) the squares centred in [x0, x0+s] x [y0, y0+s]."""
    m = (p[:, 0] >= x0) & (p[:, 0] < x0 + s) & (p[:, 1] >= y0) & (p[:, 1] < y0 + s)
    q = p.copy()
    loc = np.c_[p[m, 0] - x0, p[m, 1] - y0, p[m, 2]]
    loc = dihedral(loc, s, k)
    q[m] = np.c_[loc[:, 0] + x0, loc[:, 1] + y0, loc[:, 2]]
    return q, int(m.sum())


def settle(p, L, rounds=25):
    """Flatten a turned layout into a feasible packing: (packing, side) or None."""
    r = compact_squeeze(p[:, 0], p[:, 1], p[:, 2], L * 1.002, rounds=rounds)
    if r is None:
        return None
    x, y, t, _ = r
    q = normalize(np.c_[x, y, t])
    return q, side_of(q)


def random_move(p, L, rng, faces, kinds=("slice", "block")):
    """One random cube move on packing p (side L). faces: list of packings in the same [0, L] frame.

    Returns (layout, description). The layout may hold a different number of squares.
    """
    kind = kinds[rng.integers(len(kinds))]
    if kind == "slice":
        axis = int(rng.integers(2))
        fi = int(rng.integers(len(faces)))
        other = faces[fi]
        cs, c1 = straddle_profile(p, axis, L)
        _, c2 = straddle_profile(other, axis, L)
        cnt = c1 + c2
        width = rng.uniform(1.0, max(1.5, L / 2))
        a = pick_cut(cs, cnt, rng, -0.01, L - width)
        if a is None:
            return None, None
        b = pick_cut(cs, cnt, rng, a + 0.8 * width, a + 1.2 * width)
        if b is None:
            return None, None
        if a < 0.05:
            a = -1.0
        if b > L - 0.05:
            b = L + 1.0
        return slice_turn(p, other, axis, a, b), f"slice ax{axis} face{fi} [{a:.2f},{b:.2f})"
    s = rng.uniform(1.5, max(2.0, L / 2))
    x0, y0 = rng.uniform(0, L - s, size=2)
    k = int(rng.integers(1, 8))
    q, m = block_turn(p, x0, y0, s, k)
    return q, f"block k{k} s{s:.2f} at ({x0:.2f},{y0:.2f}) m{m}"
