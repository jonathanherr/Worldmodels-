"""Block (cluster) compositions: build a packing for n from stored packings of smaller m.

The L x L square is cut into horizontal shelves. Each shelf of height h is filled left to right
with items:

  * a block: the stored packing of m (side s(m) <= h) with the s(m) x (h - s(m)) gap above it
    filled by a unit grid or a smaller block plus grid (morph.rect_count), or
  * a unit-grid column of width 1 holding floor(h) squares.

So one packing mixes blocks of many sizes (and tilts, since the blocks are record packings),
at the scale of each shelf. Choosing the items of a shelf is an unbounded knapsack over real
widths, solved once per shelf height as W_h[v] = least width that holds >= v squares. Stacking
shelves is a second knapsack over heights for a given side L. This includes every 2x2
composition of wm/morph.py (two shelves of heights s_a and s_b) and the one-block L-extensions.
"""

import numpy as np

from . import morph

GAP = morph.BLOCK_GAP  # between items and between shelves, so squares never touch exactly


class Blocks:
    """Stored sides made monotone (a packing of k >= m with squares dropped serves m)."""

    def __init__(self, raw_sides, packings):
        """raw_sides[m] for m = 0..N (raw_sides[0] = 0); packings(k) -> (x, y, t, L) in [0, L]^2."""
        raw = np.asarray(raw_sides, dtype=float)
        n = len(raw) - 1
        src = np.arange(n + 1)
        for m in range(n - 1, 0, -1):
            if raw[src[m + 1]] < raw[src[m]]:
                src[m] = src[m + 1]
        self.sides, self.src, self.n = raw[src], src, n
        self._packings = packings
        # frontier: the largest m for each distinct side
        self.frontier = [m for m in range(1, n + 1) if m == n or self.sides[m] < self.sides[m + 1] - 1e-12]

    def packing(self, m):
        x, y, t, L = self._packings(int(self.src[m]))
        return x[:m], y[:m], t[:m], L

    def rect_count(self, w, h):
        if w < 1 or h < 1:
            return 0
        return morph.rect_count(w, h, self.sides)

    def rect_fill(self, w, h, x0, y0):
        if w < 1 or h < 1:
            return np.zeros(0), np.zeros(0), np.zeros(0)
        return morph.rect_fill(w, h, self.sides, self.packing, x0, y0)


class ShelfTable:
    """W[v] = least width of a shelf of height h holding >= v squares, with the item choices."""

    def __init__(self, blocks: Blocks, h: float, vmax: int):
        self.h = h
        items = [("col", 0, 1.0 + GAP, int(np.floor(h + 1e-9)))]
        for m in blocks.frontier:
            s = blocks.sides[m]
            if s > h + 1e-12:
                break
            items.append(("block", m, s + GAP, m + blocks.rect_count(s, h - s - GAP)))
        self.items = items
        w = np.array([it[2] for it in items])
        val = np.array([it[3] for it in items])
        keep = val > 0
        self.items = [it for it, k in zip(items, keep) if k]
        w, val = w[keep], val[keep]
        W = np.full(vmax + 1, np.inf)
        choice = np.full(vmax + 1, -1)
        W[0] = 0.0
        for v in range(1, vmax + 1):
            c = W[np.maximum(v - val, 0)] + w
            i = int(np.argmin(c))
            W[v], choice[v] = c[i], i
        self.W, self.choice = W, choice

    def capacity(self, L):
        """Most squares a shelf of width L holds."""
        return int(np.searchsorted(self.W, L, side="right")) - 1

    def items_for(self, v):
        out = []
        while v > 0:
            it = self.items[self.choice[v]]
            out.append(it)
            v = max(v - it[3], 0)
        return out


class BlockPacker:
    def __init__(self, blocks: Blocks, max_side: float | None = None, vmax: int | None = None):
        self.blocks = blocks
        vmax = vmax or blocks.n
        max_side = max_side or blocks.sides[blocks.n] + 1
        heights = sorted({float(k) for k in range(1, int(max_side) + 1)} |
                         {float(blocks.sides[m]) for m in blocks.frontier if blocks.sides[m] <= max_side})
        self.shelves = [ShelfTable(blocks, h, vmax) for h in heights]
        self.heights = np.array([t.h for t in self.shelves]) + GAP

    def capacity(self, width, height, vmax=None):
        """Most squares shelves of the given width stack to within the given height."""
        vmax = vmax or self.blocks.n
        g = np.array([t.capacity(width) for t in self.shelves])
        ok = (g > 0) & (self.heights <= height)
        if not ok.any():
            return 0
        g, hs = g[ok], self.heights[ok]
        H = np.full(vmax + 1, np.inf)
        H[0] = 0.0
        for v in range(1, vmax + 1):
            H[v] = (H[np.maximum(v - g, 0)] + hs).min()
            if H[v] > height:
                return v - 1
        return vmax

    def plan(self, L, n, height=None):
        """Shelves (table, count) of width L stacking to height <= L (or height) with >= n squares, or None."""
        height = L if height is None else height
        g = np.array([t.capacity(L) for t in self.shelves])
        ok = g > 0
        if not ok.any():
            return None
        g, hs, idx = g[ok], self.heights[ok], np.flatnonzero(ok)
        H = np.full(n + 1, np.inf)
        choice = np.full(n + 1, -1)
        H[0] = 0.0
        for v in range(1, n + 1):
            c = H[np.maximum(v - g, 0)] + hs
            i = int(np.argmin(c))
            H[v], choice[v] = c[i], i
        if H[n] > height:
            return None
        out, v = [], n
        while v > 0:
            i = choice[v]
            out.append((self.shelves[idx[i]], int(g[i])))
            v = max(v - g[i], 0)
        return out

    def min_side(self, n, hi, lo=None, tol=1e-7):
        """Smallest L in [lo, hi] (bisection) with a plan for n, or None if hi fails."""
        if self.plan(hi, n) is None:
            return None
        lo = lo if lo is not None else np.sqrt(n)
        while hi - lo > tol:
            mid = 0.5 * (lo + hi)
            if self.plan(mid, n) is None:
                lo = mid
            else:
                hi = mid
        return hi

    def build(self, L, n):
        """Packing (x, y, t) of exactly n squares from plan(L, n), or None."""
        plan = self.plan(L, n)
        if plan is None:
            return None
        xs, ys, ts = [], [], []
        y0 = 0.0
        for table, v in plan:
            h = table.h
            x0 = 0.0
            for kind, m, w, val in table.items_for(v):
                if kind == "col":
                    k = int(np.floor(h + 1e-9))
                    ex = min(GAP, (h - k) / (k + 1)) if k else 0.0
                    j = np.arange(k)
                    xs.append(np.full(k, x0 + 0.5)), ys.append(y0 + (j + 1) * ex + j + 0.5), ts.append(np.zeros(k))
                else:
                    px, py, pt, pL = self.blocks.packing(m)
                    xs.append(px + x0), ys.append(py + y0), ts.append(pt)
                    f = self.blocks.rect_fill(pL, h - pL - GAP, x0, y0 + pL + GAP)
                    xs.append(f[0]), ys.append(f[1]), ts.append(f[2])
                x0 += w
            y0 += h + GAP
        x, y, t = np.concatenate(xs), np.concatenate(ys), np.concatenate(ts)
        return x[:n], y[:n], t[:n]
