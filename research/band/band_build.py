"""Whole-box band packings for n = k^2 + m, then SLP polish of the whole box.

Design (Couzo n1765 type): box k+f; a full-width bottom band of height r+f made of columns of r+1
squares tilted by phi ((r+1)cos phi + sin phi = r+f); both band ends are exact DP staircases
(axis rows + truncated columns, research/corner_scale); k-r axis rows above. Find the smallest
grid f whose best band holds >= n squares, assemble it, drop extras, SLP the whole packing.
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import sys, math, json, functools
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'research', 'slp'))
from research.corner_scale.dpend import phi_for
from research.corner_scale.seqend import Seq
from research.corner_scale.fastdp import dp as fdp
from research.corner_scale.dpbuild import squares as end_squares
from slp import slp, violation, pairs_of, side


@functools.lru_cache(maxsize=None)
def table(r, f, na=40, noff=6):
    phi = phi_for(r, f); hs = (r + f) * math.sin(phi)
    Jmax = int(math.ceil(hs)) + 6
    a0s = np.linspace(-hs - 1.5, -hs - 0.5, na, endpoint=False)
    N, how = [], []
    for a0 in a0s:
        best = (-1, None)
        for off in np.linspace(0, f, noff):
            S = Seq(r, f, a0, off, Jmax)
            for kind in ('dec', 'inc'):
                _, n = fdp(S, kind)
                if n > best[0]:
                    best = (n, (float(off), kind))
        N.append(best[0]); how.append(best[1])
    return dict(phi=phi, Jmax=Jmax, a0=a0s, N=np.array(N), how=how, c=a0s + Jmax)


def band_count(t, r, f, Lb):
    m = r + 1; H = r + f
    cs, sn = math.cos(t['phi']), math.sin(t['phi'])
    D = Lb * cs - H * sn
    M = np.floor(D - t['c'][:, None] - t['c'][None, :] + 1e-12)
    cnt = np.where(M >= 0, t['N'][:, None] + t['N'][None, :] + M * m, -1)
    i, j = np.unravel_index(np.argmax(cnt), cnt.shape)
    return int(cnt[i, j]), int(i), int(j)


def end_gadget(r, f, t, i):
    off, kind = t['how'][i]
    S = Seq(r, f, t['a0'][i], off, t['Jmax'])
    J, _ = fdp(S, kind, want_seq=True)
    sq = end_squares(S, J)
    return dict(r=r, f=f, phi=S.phi, c=S.c, x=sq[:, 0], y=sq[:, 1], t=sq[:, 2])


def full_column(r, phi, a):
    cs, sn = math.cos(phi), math.sin(phi)
    b = (a + 1) * sn / cs
    return [((a + .5) * cs + (b + i + .5) * sn, -(a + .5) * sn + (b + i + .5) * cs, -phi) for i in range(r + 1)]


def assemble(k, r, f, t, i, j):
    gL, gR = end_gadget(r, f, t, i), end_gadget(r, f, t, j)
    phi = t['phi']; cs, sn = math.cos(phi), math.sin(phi)
    L, H = k + f, r + f
    sq = list(zip(gL['x'], gL['y'], gL['t']))
    sq += [(L - x, H - y, tt) for x, y, tt in zip(gR['x'], gR['y'], gR['t'])]
    D = L * cs - H * sn - gL['c'] - gR['c']
    M = int(math.floor(D + 1e-12))
    for jj in range(M):
        sq += full_column(r, phi, gL['c'] + jj)
    for a in range(k - r):
        for b in range(k):
            sq.append((b + .5, H + a + .5, 0.0))
    return np.array(sq), L


def design(k, m, fs, rs):
    """Smallest f in fs (sorted) with a band design holding >= k^2+m squares."""
    n = k * k + m
    for f in fs:
        for r in rs:
            if r >= k:
                continue
            t = table(r, round(f, 4))
            c, i, j = band_count(t, r, f, k + f)
            if c + (k - r) * k >= n:
                return f, r, i, j
    return None


def build_and_polish(k, m, fs, rs, iters=300):
    d = design(k, m, fs, rs)
    if d is None:
        return None
    f, r, i, j = d
    t = table(r, round(f, 4))
    P, L = assemble(k, r, f, t, i, j)
    n = k * k + m
    # drop extras from the top axis rows (rightmost first)
    extra = len(P) - n
    if extra > 0:
        ax = np.nonzero(P[:, 2] == 0)[0]
        order = ax[np.lexsort((-P[ax, 0], -P[ax, 1]))]
        P = np.delete(P, order[:extra], axis=0)
    v0 = violation(P[:, 0], P[:, 1], P[:, 2], pairs_of(P[:, 0], P[:, 1]), L)
    x, y, tt, L1, v, L0 = slp(P[:, 0], P[:, 1], P[:, 2], iters=iters)
    return dict(k=k, m=m, n=n, f0=f, r=r, v0=float(v0), L0=float(L0), L=float(L1), v=float(v), xyt=np.c_[x, y, tt])


if __name__ == '__main__':
    k, m = int(sys.argv[1]), int(sys.argv[2])
    fs = np.round(np.arange(float(sys.argv[3]), float(sys.argv[4]), 0.005), 4)
    res = build_and_polish(k, m, fs, range(4, k))
    if res is None:
        print('no design'); sys.exit()
    print({a: b for a, b in res.items() if a != 'xyt'})
