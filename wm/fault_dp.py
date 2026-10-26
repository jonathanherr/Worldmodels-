"""Fault-line gadget composition for Squares in Squares.

Start from a K x K grid of unit cells and cut it along a monotone staircase running from the
top-left corner to the bottom-right one. Cells below / left of the cut stay on the integer
lattice; cells above / right of it move by (+f, +f). The container grows from K to K + f and
the grid is no fuller, but the staircase now has room for *gadgets*.

A gadget is a corner crop of a known packing: the c = w*h + d squares of the donor that stick
out least from its w x h corner rectangle, where f is how far the worst of them sticks out. Put
on the staircase in place of a w x h block of cells, a gadget's left and bottom edges meet
lattice cells and its right and top edges meet shifted cells, so it fits as long as its own
bloat is at most f. Chaining gadgets along the staircase from the top-left gives

    n = K^2 + sum(d),   side = K + max(f).

Consecutive gadgets are separated by one buffer column (mode "X"), one buffer row ("Y"), or
nothing ("0"); mode 0 needs the first gadget's bottom-right tip or the second's top-left tip to
be empty, since those two f x f corners overlap.

Pipeline: donors (any packing, any n) -> gadgets (8 symmetries x every w, h <= MAX_WH) -> DP
over chains, state (W used, H used, d, last tip free) -> min max f -> build, then check every
candidate with the official metric. The idea follows the public "fault-line gadget DP" used by
the Kaggle leaders; this is an independent implementation.
"""

import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

MAX_WH = 31
MAX_D = 64
F_MAX = 0.999
TIP = 1.0  # tip box size for mode-0 compatibility; f < 1, so this is conservative


def half_ext(a):
    """Half of a unit square's axis-aligned extent at angle a (radians)."""
    return (np.abs(np.cos(a)) + np.abs(np.sin(a))) / 2


def normalize(p):
    """Translate (x, y, angle) rows so the bounding box starts at the origin."""
    p = np.array(p, dtype=float)
    h = half_ext(p[:, 2])
    p[:, 0] -= (p[:, 0] - h).min()
    p[:, 1] -= (p[:, 1] - h).min()
    return p


def side_of(p):
    h = half_ext(p[:, 2])
    return max(np.ptp(np.r_[p[:, 0] - h, p[:, 0] + h]), np.ptp(np.r_[p[:, 1] - h, p[:, 1] + h]))


def symmetries(p):
    """The 8 dihedral images of a packing, each normalized. Angles only matter mod 90 degrees."""
    out = []
    for swap in (False, True):
        q = p[:, [1, 0, 2]] * [1, 1, -1] if swap else p
        for fx in (1, -1):
            for fy in (1, -1):
                r = q * [fx, fy, fx * fy]
                out.append(normalize(r))
    return out


# ----------------------------------------------------------------------------- donors


def load_cert(path):
    """A franciscouzo/square-packing certificate: 'n side' then 'x y tan(theta/2)' per square."""
    lines = [ln.split() for ln in Path(path).read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    n = int(lines[0][0])
    rows = []
    for a, b, t in lines[1:n + 1]:
        rows.append((float(Fraction(a)), float(Fraction(b)), 2 * math.atan(float(Fraction(t)))))
    return normalize(rows)


def load_csv_donors(df):
    """{n: packing} from a submission DataFrame (angles in radians)."""
    out = {}
    for n, g in df.groupby("n"):
        g = g.sort_values("square_id")
        out[int(n)] = normalize(g[["x", "y", "angle"]].to_numpy(float))
    return out


# ----------------------------------------------------------------------------- gadgets


@dataclass(frozen=True)
class Gadget:
    w: int
    h: int
    d: int
    f: float
    tl_free: bool
    br_free: bool
    donor: int
    sym: int


def _crop_order(q, w, h):
    he = half_ext(q[:, 2])
    r = np.maximum(np.maximum(q[:, 0] + he - w, q[:, 1] + he - h), 0)
    return r, np.argsort(r, kind="stable")


def extract(donor_id, p, max_wh=MAX_WH, max_d=MAX_D, tip=TIP):
    """Best gadget for each (w, h, d, tl_free, br_free) cut from one donor.

    tip is the size of the corner box that must be empty for a free tip; None uses the
    gadget's own f (optimistic: right when the chain's f is no larger, so check the result).
    """
    best = {}
    for s, q in enumerate(symmetries(p)):
        he = half_ext(q[:, 2])
        keep = (q[:, 0] + he < max_wh + 1) & (q[:, 1] + he < max_wh + 1)
        q, he = q[keep], he[keep]
        xlo, xhi, ylo, yhi = q[:, 0] - he, q[:, 0] + he, q[:, 1] - he, q[:, 1] + he
        m = len(q)
        if m < 2:
            continue
        hs = np.arange(1, max_wh + 1)
        for w in range(1, max_wh + 1):
            r = np.maximum(np.maximum(xhi - w, 0)[:, None], yhi[:, None] - hs[None, :])
            r = np.maximum(r, 0)
            order = np.argsort(r, axis=0, kind="stable")
            rs = np.take_along_axis(r, order, axis=0)
            for h in range(1, max_wh + 1):
                base = w * h
                if base + 1 > m:
                    break
                col = rs[:, h - 1]
                for d in range(1, min(max_d, m - base) + 1):
                    f = col[base + d - 1]
                    if f >= F_MAX:
                        break
                    if base + d < m and col[base + d] <= f + 1e-12:
                        continue  # the same f holds more squares
                    idx = order[: base + d, h - 1]
                    tb = f if tip is None else tip
                    br = not np.any((xhi[idx] > w + 1e-9) & (ylo[idx] < tb))
                    tl = not np.any((yhi[idx] > h + 1e-9) & (xlo[idx] < tb))
                    key = (w, h, d, tl, br)
                    if key not in best or f < best[key].f:
                        best[key] = Gadget(w, h, d, float(f), tl, br, donor_id, s)
    return list(best.values())


def gadget_squares(p, g):
    q = symmetries(p)[g.sym]
    r, order = _crop_order(q, g.w, g.h)
    return q[order[: g.w * g.h + g.d]]


# ----------------------------------------------------------------------------- DP


def build_groups(gadgets, max_d=MAX_D):
    """{(w, h, tl, br): (fvec over d, {d: gadget})}, with tip-free gadgets also offered as tipped."""
    groups = {}
    for g in gadgets:
        for tl in {g.tl_free, False}:
            for br in {g.br_free, False}:
                fv, src = groups.setdefault((g.w, g.h, tl, br), (np.full(max_d + 1, np.inf), {}))
                if g.f < fv[g.d]:
                    fv[g.d] = g.f
                    src[g.d] = g
    return groups


def run_dp(groups, rounds, max_wh=MAX_WH, max_d=MAX_D):
    """layers[r][b][W, H, D] = min max f over chains of r + 1 gadgets whose last br_free is b."""
    S = max_wh + 2
    shape = (S, S, max_d + 1)
    first = [np.full(shape, np.inf), np.full(shape, np.inf)]
    for (w, h, tl, br), (fv, _) in groups.items():
        np.minimum(first[br][w, h], fv, out=first[br][w, h])
    layers = [first]
    items = [(k, fv, np.nonzero(np.isfinite(fv))[0]) for k, (fv, _) in groups.items()]
    for _ in range(rounds):
        prev = layers[-1]
        nxt = [np.full(shape, np.inf), np.full(shape, np.inf)]
        for (w, h, tl, br), fv, ds in items:
            for b in (0, 1):
                src = prev[b]
                for bx, by in [(1, 0), (0, 1)] + ([(0, 0)] if (b or tl) else []):
                    dw, dh = w + bx, h + by
                    if dw >= S - 1 or dh >= S - 1:
                        continue
                    a = src[: S - dw, : S - dh]
                    tgt_all = nxt[br][dw:, dh:]
                    for d in ds:
                        tgt = tgt_all[:, :, d:]
                        np.minimum(tgt, np.maximum(a[:, :, : max_d + 1 - d], fv[d]), out=tgt)
        layers.append(nxt)
    return layers


def backtrack(layers, groups, r, b, W, H, D, f):
    """One chain [(gadget, mode)] reaching layers[r][b][W, H, D] <= f."""
    eps = 1e-15
    if r == 0:
        for (w, h, tl, br), (fv, src) in groups.items():
            if br == b and w == W and h == H and fv[D] <= f + eps:
                return [(src[D], None)]
        return None
    prev = layers[r - 1]
    for (w, h, tl, br), (fv, src) in groups.items():
        if br != b:
            continue
        for d in np.nonzero(fv <= f + eps)[0]:
            if d > D:
                continue
            for pb in (0, 1):
                for mode, bx, by in [("X", 1, 0), ("Y", 0, 1)] + ([("0", 0, 0)] if (pb or tl) else []):
                    W0, H0 = W - w - bx, H - h - by
                    if W0 < 1 or H0 < 1 or prev[pb][W0, H0, D - d] > f + eps:
                        continue
                    chain = backtrack(layers, groups, r - 1, pb, W0, H0, D - d, f)
                    if chain is not None:
                        return chain + [(src[d], mode)]
    return None


def best_chains(layers, K, D):
    """(f, r, b, W, H) of the best chain with total d = D fitting in a K x K grid, or None."""
    best = None
    for r, layer in enumerate(layers):
        for b in (0, 1):
            sub = layer[b][1:K + 1, 1:K + 1, D]
            i = np.argmin(sub)
            W, H = np.unravel_index(i, sub.shape)
            v = sub[W, H]
            if np.isfinite(v) and (best is None or v < best[0]):
                best = (float(v), r, b, W + 1, H + 1)
    return best


# ----------------------------------------------------------------------------- build


def compose(chain, donors, K):
    """Place a chain on the staircase of a K x K grid. Returns (squares as (x, y, angle) rows, f)."""
    parts = [gadget_squares(donors[g.donor], g) for g, _ in chain]
    f = 0.0
    for p, (g, _) in zip(parts, chain):
        he = half_ext(p[:, 2])
        f = max(f, (p[:, 0] + he).max() - g.w, (p[:, 1] + he).max() - g.h)
    pos = []
    for i, (g, mode) in enumerate(chain):
        if i == 0:
            ox, oy = 0, K - g.h
        else:
            pg = chain[i - 1][0]
            ox = pos[-1][0] + pg.w + (mode == "X")
            oy = pos[-1][1] - g.h - (mode == "Y")
        pos.append((ox, oy))
    # t[i]: cells in column i with row < t[i] stay on the lattice, the rest are shifted
    t = np.zeros(K, dtype=int)
    removed = np.zeros((K, K), dtype=bool)
    col = 0
    for i, ((ox, oy), (g, _)) in enumerate(zip(pos, chain)):
        t[col:ox] = pos[i - 1][1] if i else K
        t[ox:ox + g.w] = oy
        removed[ox:ox + g.w, oy:oy + g.h] = True
        col = ox + g.w
    t[col:] = 0
    sq = []
    for i in range(K):
        for j in range(K):
            if removed[i, j]:
                continue
            s = 0.0 if j < t[i] else f
            sq.append((i + 0.5 + s, j + 0.5 + s, 0.0))
    for p, (ox, oy) in zip(parts, pos):
        sq.extend(p + [ox, oy, 0.0])
    return np.array(sq), f


def to_rows(n, sq, dilate=1e-12):
    """Submission rows (decimal strings) with centres spread by 1 + dilate so contacts become gaps."""
    sq = normalize(sq)
    sq[:, :2] *= 1 + dilate
    return [{"id": k, "n": n, "square_id": k, "x": repr(float(sq[k, 0])), "y": repr(float(sq[k, 1])),
             "angle": repr(float(sq[k, 2]))} for k in range(n)]


def check_rows(n, rows):
    """Official metric on one puzzle: (side or None, reason)."""
    from . import kaggle_metric

    _, _, side, reason = kaggle_metric.evaluate_single_puzzle(n, rows)
    return (float(side) if side else None), reason


def drop_to(sq, n):
    """Keep n squares, dropping the ones nearest the top-right corner (keeps the side)."""
    he = half_ext(sq[:, 2])
    key = np.maximum(sq[:, 0] + he, sq[:, 1] + he)
    return sq[np.sort(np.argsort(key, kind="stable")[:n])] if len(sq) > n else sq
