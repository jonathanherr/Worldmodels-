"""Backtrack the monotone staircase DP into explicit squares, then compact the end gadget."""
import math, sys, json
import numpy as np
sys.path.insert(0, '/home/claude/worldmodels-')
from research.corner_scale.dpend import phi_for, col_fit
from research.corner_scale.seqend import Seq
from research.corner_scale.endgadget import Gadget

def dp_seq(S, kind):
    R, Jmax = S.R, S.Jmax
    NEG = -10**9
    rowv = S.rowv
    cs, sn, H, off, m, a = S.cs, S.sn, S.H, S.off, S.m, S.a
    if kind == 'dec':
        def g(j, lo):
            if lo >= R: return 0
            return col_fit(a[j], 0.0 if lo == 0 else off + lo, H, cs, sn, m)
        order = range(R); first = 0
    else:
        def g(j, hi):
            if hi <= 0: return 0
            return col_fit(a[j], 0.0, H if hi == R else off + hi, cs, sn, m)
        order = range(R - 1, -1, -1); first = R - 1
    best = np.full((R, Jmax + 1), NEG); arg = np.full((R, Jmax + 1), -1)
    for j in range(Jmax + 1):
        best[first][j] = rowv[first][j] + sum(g(jj, 0 if kind == 'dec' else R) for jj in range(j, Jmax))
    prev = first
    for i in list(order)[1:]:
        bnd = i if kind == 'dec' else i + 1
        for j in range(Jmax + 1):
            for jp in range(j, Jmax + 1):
                if best[prev][jp] == NEG: continue
                v = best[prev][jp] + rowv[i][j] + sum(g(jj, bnd) for jj in range(j, jp))
                if v > best[i][j]:
                    best[i][j], arg[i][j] = v, jp
        prev = i
    last = list(order)[-1]
    j = int(np.argmax(best[last])); N = int(best[last][j])
    J = np.zeros(R, int); J[last] = j
    seq = list(order)
    for idx in range(len(seq) - 1, 0, -1):
        i = seq[idx]; J[seq[idx - 1]] = arg[i][J[i]]
    return J, N

def squares(S, J):
    sq = []
    for i in range(S.R):
        for q in range(S.rowv[i][J[i]]):
            sq.append((q + 0.5, S.off + i + 0.5, 0.0))
    cs, sn = S.cs, S.sn
    for j in range(int(J.min()), S.Jmax):
        cov = J <= j; i = 0
        while i < S.R:
            if cov[i]:
                s0 = i
                while i < S.R and cov[i]: i += 1
                y0 = 0.0 if s0 == 0 else S.off + s0
                y1 = S.H if i == S.R else S.off + i
                n = col_fit(S.a[j], y0, y1, cs, sn, S.m)
                a = S.a[j]
                yw = -(a + 1) * sn - a * cs * cs / sn
                y0e = max(y0, yw)
                b = (y0e + (a + 1) * sn) / cs
                for t in range(n):
                    u, v = a + 0.5, b + t + 0.5
                    sq.append((u * cs + v * sn, -u * sn + v * cs, -S.phi))
            else:
                i += 1
    return np.array(sq)

def best_dp(r, f, na=41, noff=9):
    phi = phi_for(r, f); hs = (r + f) * math.sin(phi)
    Jmax = int(math.ceil(hs)) + 6
    best = None
    for a0 in np.linspace(-hs - 1.5, -hs - 0.5, na):
        for off in np.linspace(0, f, noff):
            S = Seq(r, f, a0, off, Jmax)
            for kind in ('dec', 'inc'):
                J, N = dp_seq(S, kind)
                w = S.waste(N)
                if best is None or w < best[0]:
                    best = (w, S, J, N, kind)
    return best

if __name__ == '__main__':
    r, f = int(sys.argv[1]), float(sys.argv[2]); out = sys.argv[3]
    w, S, J, N, kind = best_dp(r, f)
    sq = squares(S, J)
    G = Gadget(r, f, S.phi)
    x, y, t = sq[:, 0], sq[:, 1], sq[:, 2]
    print(f'DP waste {w:.4f} per row {w/S.H:.4f} N={N} len={len(sq)} kind={kind} J={J.tolist()} feasible={G.feasible(x,y,t,S.c+1e-9)}', flush=True)
    cx, cy, ct, cc = G.compact(x, y, t, S.c)
    res = G.squeeze(cx, cy, ct, cc + 0.002)
    if res:
        x2, y2, t2, c2 = res
        w2 = G.waste(len(x2), c2)
        print(f'compacted waste {w2:.4f} per row {w2/S.H:.4f}', flush=True)
        json.dump({'r': r, 'f': f, 'phi': S.phi, 'N': len(x2), 'c': c2, 'waste': w2, 'x': x2.tolist(), 'y': y2.tolist(), 't': t2.tolist()}, open(out, 'w'))
    json.dump({'r': r, 'f': f, 'phi': S.phi, 'N': len(x), 'c': S.c, 'waste': w, 'x': x.tolist(), 'y': y.tolist(), 't': t.tolist()}, open(out.replace('.json', '_dp.json'), 'w'))
