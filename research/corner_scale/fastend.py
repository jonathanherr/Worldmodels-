"""Fast end-gadget search for a tilted-column band (left wall at x=0).

Columns j = 0..J-1 are adjustable: keep squares lo..hi-1, slid by `slide` squares along the column.
Columns J.. are full. Axis rows of height 1 at y = off + i fill from the wall to the first blocker.
Score: end waste e = m (c + s H / 2) - N, c = u-position of the left face of column J.
"""
import math, sys
import numpy as np

class End:
    def __init__(self, r, f, phi, X0, J):
        self.r, self.f, self.phi, self.X0, self.J = r, f, phi, X0, J
        self.m = r + 1; self.H = r + f
        self.cs, self.sn = math.cos(phi), math.sin(phi)
        self.p = 1 / self.cs
        # left face of column J in u-coordinates
        aJ = self._col_a(J)
        self.c = aJ
        # blocker columns J..J+2 squares (full)
        self.blk = np.array([q for j in range(J, J + 3) for q in self.col(j, 0, self.m, 0.0)])

    def _col_a(self, j):
        # column with bottom-square centre x = X0 + j p ; its u-range start a: centre u = a + 0.5
        X = self.X0 + j * self.p
        bx, by = X, (self.sn + self.cs) / 2
        u = bx * self.cs - by * self.sn
        return u - 0.5

    def col(self, j, lo, hi, slide):
        a = self._col_a(j)
        b = (a + 1) * self.sn / self.cs
        out = []
        for i in range(lo, hi):
            u, v = a + 0.5, b + i + slide + 0.5
            out.append((u * self.cs + v * self.sn, -u * self.sn + v * self.cs))
        return out

    def leftx(self, C, y0, y1):
        """min x of squares (centres C, angle -phi) within slab [y0,y1]; inf if none."""
        if len(C) == 0:
            return np.inf
        cs, sn = self.cs, self.sn
        cx, cy = C[:, 0], C[:, 1]
        # vertices for rotation -phi: offsets dx*(c,-s) + dy*(s,c)
        # bottom: (-.5,-.5)->? compute all 4
        offs = [(-.5, -.5), (.5, -.5), (.5, .5), (-.5, .5)]
        V = [(cx + dx * cs + dy * sn, cy - dx * sn + dy * cs) for dx, dy in offs]
        # left vertex = (-.5,+.5)?  find by min x offset
        # left chain: bottom vertex -> left vertex -> top vertex
        xs = np.stack([v[0] for v in V], 1); ys = np.stack([v[1] for v in V], 1)
        ib = np.argmin(ys[0]); it = np.argmax(ys[0]); il = np.argmin(xs[0])
        B = (xs[:, ib], ys[:, ib]); T = (xs[:, it], ys[:, it]); Lv = (xs[:, il], ys[:, il])
        def xl(y):
            # left chain x at height y (assumes B.y <= y <= T.y)
            t1 = np.clip((y - B[1]) / np.maximum(Lv[1] - B[1], 1e-12), 0, 1)
            t2 = np.clip((y - Lv[1]) / np.maximum(T[1] - Lv[1], 1e-12), 0, 1)
            return np.where(y <= Lv[1], B[0] + t1 * (Lv[0] - B[0]), Lv[0] + t2 * (T[0] - Lv[0]))
        hit = (T[1] > y0) & (B[1] < y1)
        if not hit.any():
            return np.inf
        ya = np.clip(y0, B[1], T[1]); yb = np.clip(y1, B[1], T[1])
        cand = np.minimum(xl(ya), xl(yb))
        inl = (Lv[1] > y0) & (Lv[1] < y1)
        cand = np.where(inl, np.minimum(cand, Lv[0]), cand)
        return cand[hit].min()

    def evaluate(self, params, off):
        cs, sn = self.cs, self.sn
        C = []
        for j, (lo, hi, sl) in enumerate(params):
            for (x, y) in self.col(j, lo, hi, sl):
                # wall: min x of square = x - (cs+sn)/2 ; floor/ceiling
                if x - (cs + sn) / 2 < -1e-12 or y - (cs + sn) / 2 < -1e-12 or y + (cs + sn) / 2 > self.H + 1e-12:
                    continue
                C.append((x, y))
        C = np.array(C) if C else np.zeros((0, 2))
        allC = np.vstack([C, self.blk]) if len(C) else self.blk
        nrows = 0
        rows = []
        i = 0
        while off + i + 1 <= self.H + 1e-12:
            lim = self.leftx(allC, off + i + 1e-9, off + i + 1 - 1e-9)
            n = int(math.floor(lim + 1e-9))
            rows.append(n)
            i += 1
        N = len(C) + sum(rows)
        w = self.m * (self.c + sn * self.H / 2) - N
        return w, N, C, rows

def overlap_free(E, params):
    """Kept column squares must not overlap each other: slid segments in one column don't overlap
    their own column; adjacent columns touch along straight lines, so any slide is fine."""
    return True

def search(r, f, phi, X0, J, iters, rng):
    E = End(r, f, phi, X0, J)
    m = E.m
    params = [[0, m, 0.0] for _ in range(J)]
    offs = np.linspace(0, f, 5)
    best = None
    for off in offs:
        cur = [p[:] for p in params]
        w, *_ = E.evaluate(cur, off)
        T = 1.0
        bw, bp = w, [p[:] for p in cur]
        for it in range(iters):
            T = 1.0 * (1 - it / iters) + 0.02
            trial = [p[:] for p in cur]
            j = rng.integers(J)
            lo = int(rng.integers(0, m + 1)); hi = int(rng.integers(lo, m + 1))
            mode = rng.integers(3)
            sl = 0.0 if mode == 0 else (-lo if mode == 1 else m - hi)
            if rng.random() < 0.3:
                sl += rng.choice([-1/3, 1/3, -0.5, 0.5, -0.25, 0.25])
            trial[j] = [lo, hi, sl]
            tw, *_ = E.evaluate(trial, off)
            if tw < w or rng.random() < math.exp(-(tw - w) / T):
                cur, w = trial, tw
                if w < bw - 1e-9:
                    bw, bp = w, [p[:] for p in cur]
        if best is None or bw < best[0]:
            best = (bw, bp, off)
    return E, best

if __name__ == '__main__':
    from scipy.optimize import brentq
    r, f, J, iters = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    phi = brentq(lambda p: (r + 1) * math.cos(p) + math.sin(p) - (r + f), 1e-9, 1.4)
    rng = np.random.default_rng(0)
    p = 1 / math.cos(phi)
    for X0 in np.linspace(0, p, 7)[:-1] + 0.3:
        E, (bw, bp, off) = search(r, f, phi, X0, J, iters, rng)
        print(f'X0={X0:.3f} waste={bw:.3f} per row {bw/E.H:.3f} off={off:.2f}', flush=True)
