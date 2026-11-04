"""Exact ("coincidence") end tables for the tilted-column band.

The grid tables in research/corner_scale sample the column phase a0 on 80 points and the axis-row
offset on 8. Counts are step functions of a0; the cheapest phase for each count is exactly a
contact event (a square corner touching a row line, a column face touching the wall), i.e. a
coincidence site. Here we evaluate the DP at every such breakpoint in a0 (exact) and on a finer
offset grid, so each end reports the true Pareto front (count N, face position c)."""
import math
import numpy as np
from research.corner_scale.dpend import phi_for
from research.corner_scale.seqend import Seq
from research.corner_scale.fastdp import dp as fdp


def a0_breakpoints(r, f, off, Jmax, lo, hi):
    phi = phi_for(r, f); cs, sn = math.cos(phi), math.sin(phi)
    H = r + f; R = int(math.floor(H - off + 1e-9))
    bps = []
    # row blocks: (a0 + j + (off+i) sn)/cs = t
    for i in range(R):
        for j in range(Jmax + 1):
            base = -j - (off + i) * sn
            t0 = math.ceil((lo - base) / cs) - 1
            t = t0
            while True:
                a = t * cs + base
                if a >= hi: break
                if a >= lo: bps.append(a)
                t += 1
    # column fits: y0 candidates are 0, off+i ; y1 candidates H, off+i
    ys0 = [0.0] + [off + i for i in range(R)]
    ys1 = [H] + [off + i for i in range(1, R + 1)]
    for j in range(Jmax + 1):
        for y0 in ys0:   # yw == y0  -> a = -(y0 + sn) sn
            a = -(y0 + sn) * sn - j
            if lo <= a < hi: bps.append(a)
        for y1 in ys1:   # (y1 - yw - sn)/cs = t -> yw = y1 - sn - t cs
            for t in range(0, r + 2):
                yw = y1 - sn - t * cs
                a = -(yw + sn) * sn - j
                if lo <= a < hi: bps.append(a)
    return np.unique(np.array(bps + [lo]))


def exact_table(r, f, noff=24):
    """Pareto front of (N, a0): for each achievable N the smallest a0."""
    phi = phi_for(r, f); hs = (r + f) * math.sin(phi)
    Jmax = int(math.ceil(hs)) + 6
    lo, hi = -hs - 1.5, -hs - 0.5
    best = {}  # N -> min a0
    for off in np.linspace(0, f, noff):
        for a0 in a0_breakpoints(r, f, off, Jmax, lo, hi):
            S = Seq(r, f, a0 + 1e-11, off, Jmax)
            for kind in ('dec', 'inc'):
                _, N = fdp(S, kind)
                if N not in best or a0 < best[N][0]:
                    best[N] = (float(a0), float(off), kind)
    # Pareto: increasing N must have increasing a0; drop dominated
    items = sorted(best.items())
    front = []
    for N, v in items:
        while front and front[-1][1][0] >= v[0]:
            front.pop()
        front.append((N, v))
    # also allow wrapping: a0 + 1 adds one full column (m squares) at cost 1 -- band_count handles via M
    return {'phi': phi, 'Jmax': Jmax, 'front': front}


def band_max(tab, r, f, Lb):
    """Max squares in a band of width Lb using exact fronts at both ends."""
    m = r + 1; H = r + f
    cs, sn = math.cos(tab['phi']), math.sin(tab['phi'])
    D = Lb * cs - H * sn
    fr = tab['front']
    N = np.array([x[0] for x in fr]); c = np.array([x[1][0] for x in fr]) + tab['Jmax']
    M = np.floor(D - c[:, None] - c[None, :] + 1e-12)
    cnt = np.where(M >= 0, N[:, None] + N[None, :] + M * m, -1)
    i, j = np.unravel_index(np.argmax(cnt), cnt.shape)
    return int(cnt[i, j]), (fr[i], fr[j], int(M[i, j]))
