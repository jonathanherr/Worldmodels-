"""Find the binding gadgets of the best fault-DP chain at every (K, D) that covers an n where we trail
the leaders, polish them as rectangles with SLP + hops, and save them as extra donors."""
import sys, os, json, time, math
import numpy as np, pandas as pd
from multiprocessing import Pool
from pathlib import Path
sys.path.insert(0, '/home/claude/worldmodels-/research/slp'); sys.path.insert(0, '/home/claude/worldmodels-')
from wm import fault_dp as fd
from gadget import hop

DON = {}
def _extract(i):
    return fd.extract(i, DON[i])
def _polish(job):
    key, q, w, h, f0 = job
    t0 = time.time()
    L, b = hop(q, w, h, rounds=HOPS, verbose=False)
    return key, w, h, f0, L, b, time.time() - t0

if __name__ == '__main__':
    base, extras_glob, out, top, procs, HOPS = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    df = pd.read_csv(base)
    donors = list(fd.load_csv_donors(df).values())
    sides = {n: fd.side_of(p) for n, p in fd.load_csv_donors(df).items()}
    for path in sorted(Path('/home/claude/couzo/certificates/best').glob('n*.cert')):
        if int(path.stem[1:]) >= 1001:
            donors.append(fd.load_cert(path))
    import glob
    for pth in sorted(glob.glob(extras_glob)):
        donors.append(fd.normalize(np.load(pth)))
    DON.update(enumerate(donors))
    with Pool(procs) as pool:
        gadgets = [g for part in pool.imap_unordered(_extract, range(len(donors)), chunksize=4) for g in part]
    groups = fd.build_groups(gadgets)
    layers = fd.run_dp(groups, 4)
    print(f'DP ready {time.time()-t0:.0f}s', flush=True)
    lb = pd.read_csv('/mnt/project-files/squares/leaderboard-20261103/gap_vs_best_per_n.csv')
    best_L = dict(zip(lb.n, lb.best_L))
    value = {}
    for K in range(10, 32):
        for D in range(1, min(fd.MAX_D, 1000 - K * K) + 1):
            res = fd.best_chains(layers, K, D)
            if res is None:
                continue
            f, r, b, W, H = res
            # n this (K, D) chain is best for and where we trail
            ns = [n for n in range(K * K + 1, K * K + D + 1) if abs(sides[n] - (K + f)) < 1e-7 and sides[n] > best_L[n] + 1e-6]
            if not ns:
                continue
            chain = fd.backtrack(layers, groups, r, b, W, H, D, f)
            if not chain:
                continue
            v = sum((sides[n] - best_L[n]) / math.sqrt(n) for n in ns)
            for g, _ in chain:
                if g.f >= f - 1e-9:
                    key = (g.donor, g.sym, g.w, g.h, g.d)
                    value[key] = value.get(key, 0) + v
    keys = sorted(value, key=value.get, reverse=True)[:top]
    print(len(value), 'binding gadgets; polishing', len(keys), flush=True)
    jobs = []
    for key in keys:
        dn, sym, w, h, d = key
        g = fd.Gadget(w, h, d, 0.0, False, False, dn, sym)
        q = fd.gadget_squares(donors[dn], g)
        he = fd.half_ext(q[:, 2]); f0 = max((q[:, 0] + he).max() - w, (q[:, 1] + he).max() - h)
        jobs.append((key, q, w, h, f0))
    with Pool(procs) as pool:
        for key, w, h, f0, L, b, dt in pool.imap_unordered(_polish, jobs):
            print(key, 'value %.5f f %.6f -> %.6f %.0fs' % (value[key], f0, L, dt), flush=True)
            if L < f0 - 1e-7:
                b = b.copy(); b[:, :2] *= 1 + 3e-9
                np.save(f'{out}/c_{key[0]}_{key[1]}_{w}_{h}_{key[4]}.npy', b)
