"""General end search: any blocking sequence J(i); column j has one segment per run of rows with
J(i) <= j. Simulated annealing over sequences, seeded from the monotone DP."""
import math, sys
import numpy as np
from research.corner_scale.dpend import phi_for, col_fit

class Seq:
    def __init__(self, r, f, a0, off, Jmax):
        self.r, self.f, self.a0, self.off, self.Jmax = r, f, a0, off, Jmax
        self.m = r + 1; self.H = r + f
        self.phi = phi_for(r, f); self.cs, self.sn = math.cos(self.phi), math.sin(self.phi)
        self.R = int(math.floor(self.H - off + 1e-9))
        self.a = np.array([a0 + j for j in range(Jmax + 1)])
        i = np.arange(self.R)[:, None]
        self.rowv = np.floor(np.maximum(0.0, (self.a[None, :] + (off + i) * self.sn) / self.cs) + 1e-9).astype(int)
        self.c = self.a[Jmax]

    def value(self, J):
        R, H, off = self.R, self.H, self.off
        N = int(self.rowv[np.arange(R), J].sum())
        for j in range(int(J.min()), self.Jmax):
            cov = J <= j
            i = 0
            while i < R:
                if cov[i]:
                    s = i
                    while i < R and cov[i]: i += 1
                    y0 = 0.0 if s == 0 else off + s
                    y1 = H if i == R else off + i
                    N += col_fit(self.a[j], y0, y1, self.cs, self.sn, self.m)
                else:
                    i += 1
        return N

    def waste(self, N):
        return self.m * (self.c + self.sn * self.H / 2) - N

def anneal(S, J0, iters, rng):
    J = J0.copy(); v = S.value(J); best = (v, J.copy())
    for it in range(iters):
        T = 1.5 * (1 - it / iters) + 0.05
        Jt = J.copy()
        k = rng.integers(3)
        if k == 0:
            Jt[rng.integers(S.R)] = rng.integers(0, S.Jmax + 1)
        elif k == 1:
            i = rng.integers(S.R); Jt[i] = np.clip(Jt[i] + rng.choice([-1, 1]), 0, S.Jmax)
        else:
            i0 = rng.integers(S.R); i1 = rng.integers(i0, S.R + 1); Jt[i0:i1] = np.clip(Jt[i0:i1] + rng.choice([-1, 1]), 0, S.Jmax)
        vt = S.value(Jt)
        if vt >= v or rng.random() < math.exp((vt - v) / T):
            J, v = Jt, vt
            if v > best[0]:
                best = (v, J.copy())
    return best

if __name__ == '__main__':
    r, f = int(sys.argv[1]), float(sys.argv[2])
    Jmax = int(sys.argv[3]); iters = int(sys.argv[4])
    rng = np.random.default_rng(int(sys.argv[5]) if len(sys.argv) > 5 else 0)
    res = []
    for a0 in np.linspace(-3, -2, 21)[:-1]:
        for off in np.linspace(0, f, 5):
            S = Seq(r, f, a0, off, Jmax)
            J0 = np.full(S.R, Jmax)
            v, J = anneal(S, J0, iters, rng)
            res.append((S.waste(v), a0, off, J.tolist()))
    res.sort()
    for w, a0, off, J in res[:3]:
        print(f'r={r} f={f} waste {w:.3f} per row {w/(r+f):.3f} a0={a0:.3f} off={off:.3f} J={J}')
