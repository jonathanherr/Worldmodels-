"""Gadget farm over many donors (run on a big machine).

For every n in a submission (non-grid packings), take the densest gadgets per (w, h) whose f is
within --slack of the donor's own f, polish each as a rectangle with SLP + hops, and collect the
improved ones (dilated 1+3e-9) into one .npz for fault_dp --extra.

    python research/slp/farm_all.py submission.csv out.npz --procs 31 --per 25 --hops 2
"""
import argparse, os, sys, time, json
import numpy as np, pandas as pd
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from wm import fault_dp as fd
from gadget import hop

def gadgets_of(args):
    n, p, per, slack = args
    L = fd.side_of(p); f0 = L - np.floor(L)
    best = {}
    for g in fd.extract(n, p, max_wh=31):
        if min(g.w, g.h) < 5 or g.f > min(f0 + slack, 0.97):
            continue
        k = (g.w, g.h)
        if k not in best or g.d > best[k].d or (g.d == best[k].d and g.f < best[k].f):
            best[k] = g
    gs = sorted(best.values(), key=lambda g: -g.d)[:per]
    return [(n, g, fd.gadget_squares(p, g)) for g in gs]

def polish(job):
    n, g, q = job
    t0 = time.time()
    try:
        L, b = hop(q, g.w, g.h, rounds=HOPS, verbose=False)
    except Exception as e:  # noqa: BLE001
        return n, g, g.f, None, time.time() - t0
    return n, g, L, (b if L < g.f - 1e-7 else None), time.time() - t0

HOPS = 2
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('csv'); ap.add_argument('out')
    ap.add_argument('--procs', type=int, default=4); ap.add_argument('--per', type=int, default=25)
    ap.add_argument('--hops', type=int, default=2); ap.add_argument('--slack', type=float, default=0.02)
    ap.add_argument('--nmin', type=int, default=100); ap.add_argument('--nmax', type=int, default=1000)
    a = ap.parse_args(); HOPS = a.hops
    df = pd.read_csv(a.csv)
    donors = {int(n): g.sort_values('square_id')[['x', 'y', 'angle']].values for n, g in df.groupby('n')}
    donors = {n: p for n, p in donors.items() if a.nmin <= n <= a.nmax and np.any(np.abs(np.sin(2 * p[:, 2])) > 1e-9)}
    print(len(donors), 'donors', flush=True)
    with Pool(a.procs) as pool:
        jobs = [j for part in pool.imap_unordered(gadgets_of, [(n, p, a.per, a.slack) for n, p in donors.items()]) for j in part]
        print(len(jobs), 'gadgets', flush=True)
        out = {}
        for i, (n, g, L, b, dt) in enumerate(pool.imap_unordered(polish, jobs, chunksize=1)):
            if b is not None:
                b = b.copy(); b[:, :2] *= 1 + 3e-9
                out[f'n{n}_{g.sym}_{g.w}_{g.h}_{g.d}'] = b
                print(f'{n} {g.w}x{g.h}+{g.d} f {g.f:.6f} -> {L:.6f} {dt:.0f}s', flush=True)
            if i % 200 == 0:
                print(f'progress {i}/{len(jobs)} kept {len(out)}', flush=True)
                np.savez(a.out, **out)
        np.savez(a.out, **out)
        print('done', len(out), flush=True)
