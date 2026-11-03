"""Tabulate end gadgets: for band height r+f and face offsets a0 over one period, the best DP
end count N(a0). Saves {(r,f): {'Jmax':.., 'a0': [...], 'N': [...], 'best': [(off, kind)]}}."""
import sys, math, json, os
import numpy as np
sys.path.insert(0, '/home/claude/worldmodels-')
from research.corner_scale.dpend import phi_for
from research.corner_scale.seqend import Seq
from research.corner_scale.fastdp import dp as fdp

def table(r, f, na=80, noff=8):
    phi = phi_for(r, f); hs = (r + f) * math.sin(phi)
    Jmax = int(math.ceil(hs)) + 6
    a0s = np.linspace(-hs - 1.5, -hs - 0.5, na, endpoint=False)
    Ns, how = [], []
    for a0 in a0s:
        best = (-1, None)
        for off in np.linspace(0, f, noff):
            S = Seq(r, f, a0, off, Jmax)
            for kind in ('dec', 'inc'):
                _, N = fdp(S, kind)
                if N > best[0]:
                    best = (N, (float(off), kind))
        Ns.append(best[0]); how.append(best[1])
    return {'Jmax': Jmax, 'a0': a0s.tolist(), 'N': Ns, 'how': how, 'phi': phi}

if __name__ == '__main__':
    rs = [int(x) for x in sys.argv[1].split(',')]
    fs = np.round(np.arange(float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])), 4)
    out = sys.argv[5]
    res = json.load(open(out)) if os.path.exists(out) else {}
    for f in fs:
        for r in rs:
            key = f'{r},{f}'
            if key in res: continue
            try:
                res[key] = table(r, float(f))
            except Exception as e:
                res[key] = None
        json.dump(res, open(out, 'w'))
        print('f', f, 'done', flush=True)
