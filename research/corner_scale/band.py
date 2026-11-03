"""Full-width tilted-column band for n = k^2 + m.

Box [0,L]^2, L = k + f. Bottom band y in [0, H], H = r + f, filled with columns of r+1 squares
tilted by phi (column axis leans right), where (r+1) cos(phi) + sin(phi) = H. Above it k - r axis
rows of k squares. Near each wall some columns are shortened/slid and axis rows fill the gap.
"""
import math
import numpy as np
from scipy.optimize import brentq

def phi_for(r, f):
    return brentq(lambda p: (r + 1) * math.cos(p) + math.sin(p) - (r + f), 1e-9, 1.4)

def square_poly(cx, cy, a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[cx + dx * c - dy * s, cy + dx * s + dy * c] for dx, dy in ((-.5, -.5), (.5, -.5), (.5, .5), (-.5, .5))])

def slab_xrange(poly, y0, y1):
    """(min x, max x) of convex polygon clipped to y0 <= y <= y1, or None."""
    pts = []
    m = len(poly)
    for i in range(m):
        p, q = poly[i], poly[(i + 1) % m]
        if y0 <= p[1] <= y1:
            pts.append(p[0])
        for yy in (y0, y1):
            if (p[1] - yy) * (q[1] - yy) < 0:
                t = (yy - p[1]) / (q[1] - p[1])
                pts.append(p[0] + t * (q[0] - p[0]))
    if not pts:
        return None
    return min(pts), max(pts)

class Band:
    def __init__(self, r, f, L):
        self.r, self.f, self.L = r, f, L
        self.m = r + 1
        self.H = r + f
        self.phi = phi_for(r, f)
        self.c, self.s = math.cos(self.phi), math.sin(self.phi)
        self.p = 1 / self.c
        self.a = -self.phi  # rotation angle of squares (edges along (c,-s) and (s,c))

    def column(self, X, lo=0, hi=None, slide=0.0):
        """Squares lo..hi-1 of a column whose full-length bottom square centre is at x = X.
        slide moves kept squares along the column axis (in units of squares)."""
        hi = self.m if hi is None else hi
        c, s = self.c, self.s
        base_y = (s + c) / 2
        out = []
        for i in range(lo, hi):
            t = i + slide
            out.append((X + t * s, base_y + t * c, self.a))
        return out

def fill_rows(blocks, H, L, side, offset, nrows):
    """Axis rows at y in [offset + j, offset + j + 1], from the wall on `side` ('L' or 'R') up to
    the nearest blocking polygon. Returns list of squares."""
    out = []
    for j in range(nrows):
        y0, y1 = offset + j, offset + j + 1
        if y1 > H + 1e-12:
            break
        lim = L if side == 'L' else 0.0
        for poly in blocks:
            rg = slab_xrange(poly, y0 + 1e-12, y1 - 1e-12)
            if rg is None:
                continue
            if side == 'L':
                lim = min(lim, rg[0])
            else:
                lim = max(lim, rg[1])
        avail = lim if side == 'L' else L - lim
        cnt = int(math.floor(avail + 1e-12))
        for i in range(cnt):
            cx = i + 0.5 if side == 'L' else L - i - 0.5
            out.append((cx, y0 + 0.5, 0.0))
    return out
