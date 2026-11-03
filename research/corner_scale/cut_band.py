"""Shrink a full-width tilted-band k^2+m record (e.g. Couzo n=1765, k=42) to a smaller box.

Width: remove c band columns (columns run along the squares' near-vertical edge) and shift the
band's right part left by c/cos(phi); remove D squares from every other row and shift right by D.
Height: remove a horizontal slab of height D from an axis-square region and shift the part above
down by D. Then compact_squeeze finds the smallest feasible box.
"""
import math, sys, json
from fractions import Fraction
import numpy as np

def load_cert(path):
    rows = [l.split() for l in open(path).read().splitlines() if l.strip() and not l.startswith('#')]
    n = int(rows[0][0]); L = float(Fraction(rows[0][1]))
    P = np.array([[float(Fraction(r[0])), float(Fraction(r[1])), 2 * math.atan(float(Fraction(r[2])))] for r in rows[1:n + 1]])
    return n, L, P

def band_mask(P, deg=71.565, tol=0.6):
    d = np.degrees(P[:, 2]) % 90
    return np.abs(d - deg) < tol

def cut(P, L, D, c, x0, slabs, band_deg=71.565):
    """Return new square array after removing width (band columns c, others D) and height D."""
    P = P.copy()
    B = band_mask(P, band_deg)
    a = math.radians(band_deg)
    cs, sn = math.cos(a), math.sin(a)          # column direction (cs, sn), normal (-sn, cs)
    q = -P[:, 0] * sn + P[:, 1] * cs           # column coordinate (integer spaced), decreases with x
    h = 1.0 / sn                                # horizontal shift per removed column
    # band y-range
    yb0, yb1 = P[B, 1].min() - 0.8, P[B, 1].max() + 0.8
    # --- width cut ---
    qb = q[B]
    # columns: q values near x0 at band mid-height
    ym = 0.5 * (yb0 + yb1)
    q0 = -x0 * sn + ym * cs
    colid = np.round(q - q0).astype(int)      # 0 at cut, negative to the right
    keep = np.ones(len(P), bool)
    shiftx = np.zeros(len(P))
    # band: remove columns colid in (-c, 0], shift columns with colid <= -c
    keep[B & (colid <= 0) & (colid > -c)] = False
    shiftx[B & (colid <= -c)] = -c * h
    # others: remove centre x in [x0, x0+D), shift x >= x0+D by -D
    O = ~B
    keep[O & (P[:, 0] >= x0) & (P[:, 0] < x0 + D)] = False
    shiftx[O & (P[:, 0] >= x0 + D)] = -D
    P[:, 0] += shiftx
    P = P[keep]
    # --- height cut: slabs [(y0, h), ...] (outside the band), total height D, applied top first
    assert abs(sum(h for _, h in slabs) - D) < 1e-9
    for y0, hh in sorted(slabs, reverse=True):
        keep = ~((P[:, 1] >= y0) & (P[:, 1] < y0 + hh))
        P[P[:, 1] >= y0 + hh, 1] -= hh
        P = P[keep]
    return P, L - D

if __name__ == '__main__':
    pass
