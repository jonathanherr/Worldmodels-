"""Row/column staircase DP for the end of a tilted-column band (left wall x = 0).

Columns j = 0, 1, ... (left faces u = a0 + j). Axis rows i = 0..R-1 occupy y in [off+i, off+i+1].
Row i is blocked by column J(i) (all columns >= J(i) cover that row); it holds
floor(x_block) squares, x_block = (a_J + (off+i) sin)/cos. Monotone J(i):
  'dec' : J non-increasing upward  -> column j covers rows [lo_j, R) up to the ceiling
  'inc' : J non-decreasing upward  -> column j covers rows [0, hi_j) down to the floor
Column squares that fit its y-interval: n cos + sin <= extent (and its face stays right of the wall).
End waste e = m (c + sin H / 2) - N with c = a0 + Jmax (face of the first untouched column).
"""
import math
import numpy as np
from scipy.optimize import brentq

def phi_for(r, f):
    return brentq(lambda p: (r + 1) * math.cos(p) + math.sin(p) - (r + f), 1e-9, 1.4)

def col_fit(a, y0, y1, cs, sn, m):
    # bottom of segment must also keep the face's lowest point at x >= 0
    yw = -(a + 1) * sn - a * cs * cs / sn
    y0 = max(y0, yw)
    n = math.floor((y1 - y0 - sn) / cs + 1e-9)
    return max(0, min(m, n))

def solve(r, f, a0, off, Jmax, kind):
    m = r + 1; H = r + f
    phi = phi_for(r, f); cs, sn = math.cos(phi), math.sin(phi)
    R = int(math.floor(H - off + 1e-9))
    a = [a0 + j for j in range(Jmax + 1)]
    def xb(i, j):
        return max(0.0, (a[j] + (off + i) * sn) / cs)
    rowv = [[math.floor(xb(i, j) + 1e-9) for j in range(Jmax + 1)] for i in range(R)]
    NEG = -10**9
    if kind == 'dec':
        # g[j][lo]: column j covering rows lo..R-1 (+ ceiling gap); lo = R means absent
        def g(j, lo):
            if lo >= R: return 0
            y0 = 0.0 if lo == 0 else off + lo
            return col_fit(a[j], y0, H, cs, sn, m)
        # DP over rows: best[i][j] = best value with J(i)=j for rows 0..i, including columns opened so far
        best = [[NEG] * (Jmax + 1) for _ in range(R)]
        arg = [[None] * (Jmax + 1) for _ in range(R)]
        for j in range(Jmax + 1):
            best[0][j] = rowv[0][j] + sum(g(jj, 0) for jj in range(j, Jmax))
        for i in range(1, R):
            for j in range(Jmax + 1):
                for jp in range(j, Jmax + 1):
                    if best[i - 1][jp] == NEG: continue
                    v = best[i - 1][jp] + rowv[i][j] + sum(g(jj, i) for jj in range(j, jp))
                    if v > best[i][j]:
                        best[i][j], arg[i][j] = v, jp
        j = int(np.argmax(best[R - 1])); N = best[R - 1][j]
    else:
        def g(j, hi):
            if hi <= 0: return 0
            y1 = H if hi == R else off + hi
            return col_fit(a[j], 0.0, y1, cs, sn, m)
        best = [[NEG] * (Jmax + 1) for _ in range(R)]
        arg = [[None] * (Jmax + 1) for _ in range(R)]
        for j in range(Jmax + 1):
            best[R - 1][j] = rowv[R - 1][j] + sum(g(jj, R) for jj in range(j, Jmax))
        for i in range(R - 2, -1, -1):
            for j in range(Jmax + 1):
                for jp in range(j, Jmax + 1):
                    if best[i + 1][jp] == NEG: continue
                    v = best[i + 1][jp] + rowv[i][j] + sum(g(jj, i + 1) for jj in range(j, jp))
                    if v > best[i][j]:
                        best[i][j], arg[i][j] = v, jp
        j = int(np.argmax(best[0])); N = best[0][j]
    c = a[Jmax]
    w = m * (c + sn * H / 2) - N
    return w, N, kind

if __name__ == '__main__':
    import sys
    r, f = int(sys.argv[1]), float(sys.argv[2])
    Jmax = int(sys.argv[3]) if len(sys.argv) > 3 else 14
    best = None
    phi = phi_for(r, f); hs = (r + f) * math.sin(phi)
    Jmax = int(math.ceil(hs)) + 6
    for a0 in np.linspace(-hs - 1.5, -hs - 0.5, 41):
        for off in np.linspace(0, f, 9):
            for kind in ('dec', 'inc'):
                w, N, _ = solve(r, f, a0, off, Jmax, kind)
                if best is None or w < best[0]:
                    best = (w, N, kind, a0, off)
    w = best[0]
    print(f'r={r} f={f} best waste {w:.3f} per row {w/(r+f):.3f} {best}')
