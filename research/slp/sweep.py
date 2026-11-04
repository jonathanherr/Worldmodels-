"""SLP-polish every packing of a submission; certify with the official metric; save the winners."""
import sys, time, json, numpy as np
from multiprocessing import Pool
sys.path.insert(0, '/home/claude/worldmodels-/research/slp'); sys.path.insert(0, '/home/claude/worldmodels-')
from slp import slp, side
from wm import kaggle_metric as km

_Z = np.load(sys.argv[5] if len(sys.argv) > 5 else '/home/claude/work/cur.npz'); D = {k: _Z[k] for k in _Z.files}

def certify(n, x, y, t):
    for dil in (1e-12, 1e-10, 1e-9, 1e-8, 1e-7, 3e-7, 1e-6, 1e-5):
        h = (np.abs(np.cos(t)) + np.abs(np.sin(t))) / 2
        xx = (x - (x - h).min()) * (1 + dil) + dil; yy = (y - (y - h).min()) * (1 + dil) + dil
        rows = [dict(n=str(n), square_id=str(i), x=repr(float(xx[i])), y=repr(float(yy[i])), angle=repr(float(t[i]))) for i in range(n)]
        _, _, L, reason = km.evaluate_single_puzzle(n, rows)
        if L is not None:
            return float(L), rows, dil
    return None, None, None

def job(n):
    a = D[str(n)]
    if np.all(np.abs(np.sin(2 * a[:, 2])) < 1e-9):
        return n, 0, 0, None, None, 0
    t0 = time.time()
    x, y, t, L, v, L0 = slp(a[:, 0], a[:, 1], a[:, 2], iters=600)
    if L0 - L < 1e-7:
        return n, L0, L, None, None, time.time() - t0
    Lc, rows, dil = certify(n, x, y, t)
    return n, L0, Lc, rows, dil, time.time() - t0

if __name__ == '__main__':
    ns = [int(s) for s in open(sys.argv[1]).read().split()] if not sys.argv[1].isdigit() else range(int(sys.argv[1]), int(sys.argv[2]) + 1)
    out = sys.argv[3]
    res = {}
    with Pool(int(sys.argv[4])) as p:
        for n, L0, L, rows, dil, dt in p.imap_unordered(job, ns):
            if not L0:
                continue
            print(n, '%.9f -> %s  %.0fs dil %s' % (L0, '%.9f' % L if L else 'FAIL', dt, dil), flush=True)
            if rows and L < L0 - 1e-7:
                res[n] = dict(L0=L0, L=L, rows=rows); json.dump(res, open(out, 'w'))
