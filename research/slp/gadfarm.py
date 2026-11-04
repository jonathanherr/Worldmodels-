"""Polish the densest gadget per (w, h) of a donor as a rectangle; save each as a donor .npy."""
import sys, os, json, time, numpy as np
from multiprocessing import Pool
sys.path.insert(0, '/home/claude/worldmodels-/research/slp'); sys.path.insert(0, '/home/claude/worldmodels-')
from wm import fault_dp as fd
from gadget import hop
from gadlist import load
P = None
def job(g):
    q = fd.gadget_squares(P, g)
    t0 = time.time()
    L, b = hop(q, g.w, g.h, rounds=HOPS, verbose=False)
    path = f'{OUT}/g_{TAG}_{g.w}_{g.h}_{g.d}.npy'
    if L < g.f - 1e-7:
        np.save(path, b)
    return g.w, g.h, g.d, g.f, L, time.time() - t0
if __name__ == '__main__':
    src, n, fmax, OUT, procs, HOPS = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4], int(sys.argv[5]), int(sys.argv[6])
    TAG = os.path.basename(src).split('.')[0] + n
    os.makedirs(OUT, exist_ok=True)
    P = load(src, n)
    best = {}
    for g in fd.extract(0, P, max_wh=31):
        if g.f < fmax and min(g.w, g.h) >= 4:
            k = (min(g.w, g.h), max(g.w, g.h))
            if k not in best or g.d > best[k].d or (g.d == best[k].d and g.f < best[k].f):
                best[k] = g
    gs = sorted(best.values(), key=lambda g: g.w * g.h)
    print(len(gs), 'gadgets', flush=True)
    with Pool(procs) as pool:
        for r in pool.imap_unordered(job, gs):
            print('%d %d %d f %.6f -> %.6f %.0fs' % r, flush=True)
