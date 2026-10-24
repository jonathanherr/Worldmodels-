"""Rectangle-decomposition packings: split the container into rectangles, pack each, compose.

The container [0, s]^2 with s = m + f (m integer, 0 < f < 1) is cut into three rectangles:

    +-------------------------------+
    |  top strip, width c1 + f      |   leaning stacks, length s
    +---------------------+---------+
    |                     | right   |
    |  axis grid          | strip   |   right strip: width c2 + f, length m - c1
    |  (m - c2) x (m - c1)| c2 + f  |
    +---------------------+---------+

The grid has integer sides, so it wastes nothing. Each strip has width c + f, where an
axis-aligned packing would waste f per unit length. Instead it holds stacks of k = c + 1 squares
leaning at a small angle a (Erdos and Graham's trick): a stack of k squares spans
k cos a + sin a across the strip, which fits into c + f once a is large enough, and stacks repeat
every 1 / cos a along the strip. That packs k cos a per unit length instead of c. The leaning
leaves triangular wedges at both ends of each strip; `greedy_fill` packs those with axis-aligned
squares anchored at the walls and with partial stacks.

Everything is in plain geometry with a small clearance `GAP` between every pair of squares, so the
result passes Kaggle's interval-arithmetic metric without polishing.
"""

from dataclasses import dataclass

import numpy as np

GAP = 2e-9  # clearance kept between squares and to the walls


# ----------------------------------------------------------------------------- geometry


def square_corners(x, y, deg):
    """(N, 4, 2) corners of unit squares centered at (x, y) rotated by deg."""
    x, y, deg = (np.atleast_1d(np.asarray(a, float)) for a in (x, y, deg))
    t = np.deg2rad(deg)
    c, s = np.cos(t), np.sin(t)
    loc = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])
    px = x[:, None] + c[:, None] * loc[:, 0] - s[:, None] * loc[:, 1]
    py = y[:, None] + s[:, None] * loc[:, 0] + c[:, None] * loc[:, 1]
    return np.stack([px, py], -1)


def _axes(deg):
    t = np.deg2rad(np.atleast_1d(deg))
    c, s = np.cos(t), np.sin(t)
    return np.stack([np.stack([c, s], -1), np.stack([-s, c], -1)], 1)  # (N, 2, 2)


def overlaps_any(cx, cy, cdeg, px, py, pdeg, gap=GAP):
    """Does the square (cx, cy, cdeg) come within `gap` of any square in (px, py, pdeg)?"""
    if len(px) == 0:
        return False
    near = (np.abs(px - cx) < 1.42 + gap) & (np.abs(py - cy) < 1.42 + gap)
    if not near.any():
        return False
    px, py, pdeg = px[near], py[near], pdeg[near]
    A = square_corners(cx, cy, cdeg)  # (1, 4, 2)
    B = square_corners(px, py, pdeg)  # (P, 4, 2)
    axes = np.concatenate([np.broadcast_to(_axes(cdeg), (len(px), 2, 2)), _axes(pdeg)], 1)  # (P, 4, 2)
    pa = np.einsum("pkd,jd->pkj", axes, A[0])  # (P, 4 axes, 4 corners)
    pb = np.einsum("pkd,pjd->pkj", axes, B)
    # Axis-aligned pairs may touch (the metric certifies those exactly); any pair with a rotated
    # square keeps the clearance.
    g = np.where((cdeg == 0) & (pdeg == 0), -1e-12, gap)[:, None]
    sep = (pa.max(-1) + g <= pb.min(-1)) | (pb.max(-1) + g <= pa.min(-1))  # (P, 4)
    return bool((~sep.any(1)).any())


def inside(x, y, deg, box, gap=GAP):
    """Is the square inside box = (x0, y0, x1, y1) with clearance gap?"""
    c = square_corners(x, y, deg)[0]
    x0, y0, x1, y1 = box
    if deg == 0:
        gap = -1e-12
    return (c[:, 0].min() >= x0 + gap and c[:, 0].max() <= x1 - gap
            and c[:, 1].min() >= y0 + gap and c[:, 1].max() <= y1 - gap)


# ----------------------------------------------------------------------------- strips


def lean_angle(w: float, k: int, gap: float = GAP) -> float | None:
    """Smallest angle a (radians) with k cos a + sin a (+ gaps) <= w: a stack of k squares fits
    across a strip of width w. None if no angle below 45 degrees works."""
    need = w - 2 * gap - (k - 1) * gap
    lo, hi = 0.0, np.pi / 4
    if k * np.cos(hi) + np.sin(hi) > need:
        return None
    if k + 0.0 <= need:
        return 0.0
    for _ in range(80):  # k cos a + sin a is decreasing on the branch we want (a > atan(1/k))
        mid = (lo + hi) / 2
        if k * np.cos(mid) + np.sin(mid) <= need:
            hi = mid
        else:
            lo = mid
    return hi


def stack_lattice(w: float, k: int, a: float, j_range, i_range, gap: float = GAP):
    """Centers of the leaning-stack lattice in strip-local coordinates (length along x, width
    along y, strip = [0, L] x [0, w]). Stack j, square i; stacks lean right by angle a.

    Stack 0's lowest corner sits on y = gap and its leftmost corner on x = gap. Squares within a
    stack are spaced 1 + gap along the stack, stacks 1 + gap apart perpendicular to it.
    """
    ca, sa = np.cos(a), np.sin(a)
    d = np.array([sa, ca])  # along the stack
    nrm = np.array([ca, -sa])  # across stacks (pointing right, slightly down)
    # Local square [0,1]x[0,1] rotated clockwise by a: corner (1, 0) is lowest at y = -sa.
    origin = np.array([gap, gap + sa])
    j, i = np.meshgrid(np.asarray(j_range), np.asarray(i_range), indexing="ij")
    j, i = j.ravel(), i.ravel()
    step_j = (1 + gap) / ca  # stacks repeat along x
    ctr = (origin[None] + (i[:, None] * (1 + gap) + 0.5) * d[None] + 0.5 * nrm[None]
           + (j[:, None] * step_j) * np.array([1.0, 0.0])[None])
    return ctr[:, 0], ctr[:, 1], j, i


def strip_packing(w: float, L: float, k: int, gap: float = GAP):
    """Full leaning stacks of k squares in the strip [0, L] x [0, w], strip-local coordinates.

    Returns (x, y, deg, a) or None if the stack does not fit. Squares are rotated by -a.
    """
    a = lean_angle(w, k, gap)
    if a is None:
        return None
    ca, sa = np.cos(a), np.sin(a)
    extent = k * sa + ca + (k - 1) * gap * sa  # x-extent of one stack
    step = (1 + gap) / ca
    nst = int(np.floor((L - 2 * gap - extent) / step + 1e-12)) + 1
    if nst <= 0:
        return np.zeros(0), np.zeros(0), np.zeros(0), a
    x, y, _, _ = stack_lattice(w, k, a, range(nst), range(k), gap)
    return x, y, np.full(len(x), -np.rad2deg(a)), a


# ----------------------------------------------------------------------------- greedy fill


def greedy_fill(px, py, pdeg, candidates, box, gap=GAP):
    """Add each candidate (x, y, deg), in order, that fits in box and clears every placed square."""
    px, py, pdeg = list(px), list(py), list(pdeg)
    ax, ay, ad = np.array(px), np.array(py), np.array(pdeg)
    for cx, cy, cd in candidates:
        if not inside(cx, cy, cd, box, gap):
            continue
        if overlaps_any(cx, cy, cd, ax, ay, ad, gap):
            continue
        px.append(cx), py.append(cy), pdeg.append(cd)
        ax, ay, ad = np.append(ax, cx), np.append(ay, cy), np.append(ad, cd)
    return np.array(px), np.array(py), np.array(pdeg)


def axis_candidates(box, region, gap=GAP):
    """Axis-aligned cells on the four lattices anchored at the corners of box, restricted to
    centers in region = (x0, y0, x1, y1)."""
    X0, Y0, X1, Y1 = box
    r0, s0, r1, s1 = region
    out = []
    nx, ny = int(X1 - X0) + 1, int(Y1 - Y0) + 1
    for fx in (0, 1):
        for fy in (0, 1):
            for i in range(nx):
                cx = X0 + 0.5 + i if fx == 0 else X1 - 0.5 - i
                if not r0 <= cx <= r1:
                    continue
                for j in range(ny):
                    cy = Y0 + 0.5 + j if fy == 0 else Y1 - 0.5 - j
                    if s0 <= cy <= s1:
                        out.append((cx, cy, 0.0))
    return out


# ----------------------------------------------------------------------------- composition


@dataclass
class Layout:
    s: float
    c1: int  # top strip width c1 + f (0 = no top strip)
    c2: int  # right strip width c2 + f (0 = no right strip)
    k1: int | None = None  # squares per stack in the top strip (default c1 + 1)
    k2: int | None = None


def _to_global_top(x, y, deg, y0):
    return x, y + y0, deg


def _to_global_right(x, y, deg, x0):
    # strip-local (along, across) -> global (across, along): a reflection in the diagonal, which
    # turns a square's angle a into 90 - a, i.e. -a up to the square's symmetry.
    return x0 + y, x, -deg


def compose(lay: Layout, fill: bool = True, gap: float = GAP, order: str = "stacks"):
    """Build the three-rectangle packing for `lay`. Returns (x, y, deg)."""
    s = lay.s
    m = int(np.floor(s))
    f = s - m
    c1, c2 = lay.c1, lay.c2
    gx, gy = m - c2, m - c1  # grid width and height
    xs, ys, ds = [], [], []
    # grid, anchored at the bottom-left corner (axis-aligned squares may touch)
    for i in range(gx):
        for j in range(gy):
            xs.append(0.5 + i), ys.append(0.5 + j), ds.append(0.0)
    if c1 > 0:
        w = c1 + f
        sp = strip_packing(w - gap, s, lay.k1 or c1 + 1, gap)
        if sp is not None:
            x, y, d, _ = sp
            x, y, d = _to_global_top(x, y, d, gy)
            xs += list(x)
            ys += list(y)
            ds += list(d)
    if c2 > 0:
        w = c2 + f
        sp = strip_packing(w - gap, gy, lay.k2 or c2 + 1, gap)
        if sp is not None:
            x, y, d, _ = sp
            x, y, d = _to_global_right(x, y, d, gx)
            xs += list(x)
            ys += list(y)
            ds += list(d)
    x, y, d = np.array(xs), np.array(ys), np.array(ds)
    if fill:
        box = (0.0, 0.0, s, s)
        # partial stacks extend each strip's lattice beyond its ends; "stacks" tries them first
        part, axis = _partial_stack_candidates(lay, gap), axis_candidates(box, box, gap)
        cands = part + axis if order == "stacks" else axis + part
        x, y, d = greedy_fill(x, y, d, cands, box, gap)
    return x, y, d


def _partial_stack_candidates(lay: Layout, gap):
    s = lay.s
    m = int(np.floor(s))
    f = s - m
    out = []
    for which, c, length in (("top", lay.c1, s), ("right", lay.c2, m - lay.c1)):
        if c <= 0:
            continue
        k = (lay.k1 if which == "top" else lay.k2) or c + 1
        a = lean_angle(c + f - gap, k, gap)
        if a is None:
            continue
        nst = int(length) + 3
        x, y, _, _ = stack_lattice(c + f - gap, k, a, range(-k - 2, nst + k), range(-1, k + 1), gap)
        d = np.full(len(x), -np.rad2deg(a))
        if which == "top":
            x, y, d = _to_global_top(x, y, d, m - lay.c1)
        else:
            x, y, d = _to_global_right(x, y, d, m - lay.c2)
        out += list(zip(x, y, d))
    return out


def count_only(lay: Layout, fill: bool = True, order: str = "stacks") -> int:
    return len(compose(lay, fill, order=order)[0])


# ----------------------------------------------------------------------------- rectangle packer


R2 = np.sqrt(2.0)


def _fill_rect(x, y, d, w, L, extra=(), gap=GAP):
    """Greedy-fill the rectangle, trying `extra` candidates before and after the axis lattices."""
    box = (0.0, 0.0, L, w)
    axis = axis_candidates(box, box, gap)
    a = greedy_fill(x, y, d, list(extra) + axis, box, gap)
    b = greedy_fill(x, y, d, axis + list(extra), box, gap) if extra else a
    return a if len(a[0]) >= len(b[0]) else b


def rect_lean(w: float, L: float, k: int, gap: float = GAP):
    """Leaning stacks of k squares across [0, L] x [0, w], ends filled greedily."""
    sp = strip_packing(w, L, k, gap)
    if sp is None:
        return None
    x, y, d, a = sp
    px, py, _, _ = stack_lattice(w, k, a, range(-k - 2, int(L) + k + 3), range(-1, k + 1), gap)
    extra = list(zip(px, py, np.full(len(px), -np.rad2deg(a))))
    return _fill_rect(x, y, d, w, L, extra, gap)


def rect_band45(w: float, L: float, rows_axis: int, phase: float = 0.0, gap: float = GAP):
    """rows_axis axis-aligned rows at the bottom, a 45-degree diamond lattice above, ends filled."""
    h = 1 / R2 + gap  # half-diagonal of a diamond plus clearance
    st = (1 + gap) / R2
    y0 = rows_axis
    xs, ys = [], []
    nr = int((w - y0) / st) + 2
    ni = int(L / (2 * st)) + 3
    for r in range(nr):
        for i in range(-1, ni):
            cx = phase + h + (r % 2) * st + 2 * i * st
            cy = y0 + h + r * st
            if cx - h >= 0 and cx + h <= L and cy + h <= w:
                xs.append(cx), ys.append(cy)
    x = np.concatenate([np.array(xs), [0.5 + i for i in range(int(L)) for _ in range(rows_axis)]])
    y = np.concatenate([np.array(ys), [0.5 + j for _ in range(int(L)) for j in range(rows_axis)]])
    d = np.concatenate([np.full(len(xs), 45.0), np.zeros(int(L) * rows_axis)])
    return _fill_rect(x, y, d, w, L, (), gap)


def rect_pack(w: float, L: float, gap: float = GAP):
    """Best packing found of unit squares in [0, L] x [0, w] (L along x). Returns (x, y, deg, name)."""
    c = int(np.floor(w + 1e-12))
    gx, gy = np.meshgrid(np.arange(int(L)) + 0.5, np.arange(c) + 0.5, indexing="ij")
    best = (gx.ravel(), gy.ravel(), np.zeros(gx.size), "axis")
    tries = []
    for k in (c + 1, c + 2):
        tries.append((lambda k=k: rect_lean(w, L, k, gap), f"lean{k}"))
    for ra in range(0, c + 1):
        if w - ra >= R2 + 2 * gap:
            for ph in (0.0, 0.3535, 0.707):
                tries.append((lambda ra=ra, ph=ph: rect_band45(w, L, ra, ph, gap), f"band45/{ra}/{ph}"))
    for fn, name in tries:
        r = fn()
        if r is not None and len(r[0]) > len(best[0]):
            best = (*r, name)
    return best


def place(rect, x0, y0, vertical=False):
    """Move a rect_pack result (local [0, L] x [0, w]) to (x0, y0); vertical turns it to run up y."""
    x, y, d = rect[:3]
    if vertical:  # reflect in the diagonal: (u, v) -> (v, u); a square's angle flips sign
        return x0 + y, y0 + x, -d
    return x0 + x, y0 + y, d


def wrap_core(core, side: float, w: float, gap: float = GAP):
    """Packing `core` (x, y, deg) in [0, side]^2 plus an L of width w: a strip w x (side + w)
    on the right and w x side on top. Returns (x, y, deg) in side + w."""
    top = rect_pack(w, side, gap)
    right = rect_pack(w, side + w, gap)
    parts = [core, place(top, 0.0, side), place(right, side, 0.0, vertical=True)]
    return tuple(np.concatenate([p[i] for p in parts]) for i in range(3))
