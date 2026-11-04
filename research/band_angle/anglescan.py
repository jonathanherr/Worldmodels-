"""Band tilt angle as a free variable.

One full-width band of tilted columns (research/corner_scale model): box L = k + f, bottom band height
H = r + f, k - r axis rows of k squares above. The corner_scale/pretrain code fixes the tilt to the tight
angle phi*(r, f) with (r+1) cos phi + sin phi = H. Here phi is free (phi >= phi* so a full column still
fits; phi < phi* only fits r squares per column). Ends are exact staircase DP ends at every phi, both ends
chosen independently over a fine a0 grid, so the column phase at the two walls is free too.

count(k, f, r, phi) = N_left + N_right + (r+1) * floor(D - cL - cR) + (k - r) k,  D = L cos phi - H sin phi.
"""
import math, sys, json
import numpy as np
from multiprocessing import Pool
from research.corner_scale.seqend import Seq
from research.corner_scale.dpend import phi_for
from research.corner_scale.fastdp import dp as fdp


class SeqPhi(Seq):
    def __init__(self, r, f, a0, off, Jmax, phi):
        self.r, self.f, self.a0, self.off, self.Jmax = r, f, a0, off, Jmax
        self.m = r + 1; self.H = r + f
        self.phi = phi; self.cs, self.sn = math.cos(phi), math.sin(phi)
        self.R = int(math.floor(self.H - off + 1e-9))
        self.a = np.array([a0 + j for j in range(Jmax + 1)])
        i = np.arange(self.R)[:, None]
        self.rowv = np.floor(np.maximum(0.0, (self.a[None, :] + (off + i) * self.sn) / self.cs) + 1e-9).astype(int)
        self.c = self.a[Jmax]


def ends(r, f, phi, na=200, noff=8):
    """Best N per face c (c = a0 + Jmax) over offsets and both staircase kinds."""
    H = r + f; hs = H * math.sin(phi)
    Jmax = int(math.ceil(hs)) + 6
    cs, Ns = [], []
    for a0 in np.linspace(-hs - 1.5, -hs - 0.5, na, endpoint=False):
        best = -1
        for off in np.linspace(0, f, noff):
            S = SeqPhi(r, f, a0, off, Jmax, phi)
            for kind in ('dec', 'inc'):
                _, N = fdp(S, kind)
                best = max(best, N)
        cs.append(a0 + Jmax); Ns.append(best)
    return np.array(cs), np.array(Ns)


def combine(cs, Ns, r, f, k, phi):
    L = k + f; H = r + f
    D = L * math.cos(phi) - H * math.sin(phi)
    M = np.floor(D - cs[:, None] - cs[None, :] + 1e-12)
    cnt = np.where(M >= 0, Ns[:, None] + Ns[None, :] + M * (r + 1), -1)
    return int(cnt.max()) + (k - r) * k


def job(args):
    r, f, dphi, ks = args
    try:
        p0 = phi_for(r, f)
    except Exception:
        return None
    phi = p0 + dphi
    cs, Ns = ends(r, f, phi)
    return r, f, dphi, {k: combine(cs, Ns, r, f, k, phi) for k in ks}


if __name__ == '__main__':
    # usage: anglescan OUT.jsonl r0 r1 f0 f1 df dphimax ndphi
    out = sys.argv[1]; r0, r1 = int(sys.argv[2]), int(sys.argv[3])
    fs = np.round(np.arange(float(sys.argv[4]), float(sys.argv[5]) + 1e-9, float(sys.argv[6])), 4)
    dps = np.linspace(0, float(sys.argv[7]), int(sys.argv[8]))
    ks = list(range(15, 32))
    jobs = [(r, float(f), float(d), ks) for r in range(r0, r1 + 1) for f in fs for d in dps]
    with Pool(4) as pool, open(out, 'w') as fh:
        for i, res in enumerate(pool.imap_unordered(job, jobs, chunksize=2)):
            if res is None: continue
            r, f, d, cnt = res
            fh.write(json.dumps({'r': r, 'f': f, 'dphi': d, 'cnt': cnt}) + '\n'); fh.flush()
