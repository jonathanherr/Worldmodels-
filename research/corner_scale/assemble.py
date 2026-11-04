"""Assemble a k^2+m packing: bottom band (left gadget, full columns, right gadget = left rotated
180 deg about the band centre) plus k - r axis rows above."""
import sys, json, math
import numpy as np
sys.path.insert(0, '/home/claude/worldmodels-')
from wm import pack_opt

def full_column(r, phi, a):
    """Squares of a full column whose u-range is [a, a+1]."""
    cs, sn = math.cos(phi), math.sin(phi)
    b = (a + 1) * sn / cs
    out = []
    for i in range(r + 1):
        u, v = a + 0.5, b + i + 0.5
        out.append((u * cs + v * sn, -u * sn + v * cs, -phi))
    return out

def assemble(g, k, gR=None):
    r, f, phi, c = g['r'], g['f'], g['phi'], g['c']
    gR = gR or g
    cs, sn = math.cos(phi), math.sin(phi)
    L, H = k + f, r + f
    sq = list(zip(g['x'], g['y'], g['t']))
    sq += [(L - x, H - y, t) for x, y, t in zip(gR['x'], gR['y'], gR['t'])]
    D = L * cs - H * sn - c - gR['c']
    M = int(math.floor(D + 1e-12))
    for j in range(M):
        sq += full_column(r, phi, c + j)
    for i in range(k - r):
        for j in range(k):
            sq.append((j + 0.5, H + i + 0.5, 0.0))
    return np.array(sq), L, D - M

if __name__ == '__main__':
    g = json.load(open(sys.argv[1]))
    for k in range(20, 32):
        P, L, slack = assemble(g, k)
        ok = pack_opt.feasible(P[:, 0], P[:, 1], P[:, 2], L, 0.0)
        print(f"k={k} L={L:.4f} count={len(P)} k^2={k*k} extra={len(P)-k*k} col slack={slack:.3f} feasible={ok}")
