"""Fast monotone staircase DP (suffix-max form) used for end tables and reconstruction."""
import math
import numpy as np
from research.corner_scale.dpend import col_fit

def gtable(S, kind):
    """g[j, b]: squares of column j when it covers rows [b, R) ('dec') or [0, b) ('inc')."""
    R, J = S.R, S.Jmax
    g = np.zeros((J + 1, R + 1), dtype=np.int64)
    for j in range(J):
        for b in range(R + 1):
            if kind == 'dec':
                if b >= R: continue
                y0 = 0.0 if b == 0 else S.off + b; y1 = S.H
            else:
                if b <= 0: continue
                y0 = 0.0; y1 = S.H if b == R else S.off + b
            g[j, b] = col_fit(S.a[j], y0, y1, S.cs, S.sn, S.m)
    return g

def dp(S, kind, want_seq=False):
    R, J = S.R, S.Jmax
    g = gtable(S, kind)
    # G[b, j] = sum_{jj < j} g[jj, b]
    G = np.zeros((R + 1, J + 2), dtype=np.int64)
    G[:, 1:] = np.cumsum(g.T, axis=1)
    rowv = S.rowv
    order = list(range(R)) if kind == 'dec' else list(range(R - 1, -1, -1))
    first = order[0]
    b0 = 0 if kind == 'dec' else R
    best = rowv[first].astype(np.int64) + (G[b0, J] - G[b0, :J + 1])
    args = []
    for i in order[1:]:
        b = i if kind == 'dec' else i + 1
        val = best + G[b, :J + 1]                      # indexed by jp
        # suffix max over jp >= j
        sm = np.maximum.accumulate(val[::-1])[::-1]
        if want_seq:
            idx = np.empty(J + 1, dtype=np.int64); cur = J
            for j in range(J, -1, -1):
                if val[j] >= val[cur]: cur = j if val[j] > val[cur] or j == J else cur
                idx[j] = cur
            # recompute argmax properly
            idx = np.array([j + int(np.argmax(val[j:])) for j in range(J + 1)])
            args.append(idx)
        best = sm - G[b, :J + 1] + rowv[i]
    j = int(np.argmax(best)); N = int(best[j])
    if not want_seq:
        return None, N
    seq = np.zeros(R, dtype=np.int64); seq[order[-1]] = j
    for k in range(len(order) - 1, 0, -1):
        seq[order[k - 1]] = args[k - 1][seq[order[k]]]
    return seq, N
