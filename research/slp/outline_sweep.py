"""Sweep block-outline gadgets over (w, h, f0, seed) on many cores.

Each job: best of 40 random clipped strip starts, hill-climb the outline for --steps, then SLP-polish
(count fixed, minimise f) and save the gadget if it is feasible. One JSON line per job.

    python research/slp/outline_sweep.py --w 8 24 --h 4 14 --f0 0.72 0.78 0.84 --seeds 2 --procs 31 \
        --out sweep.jsonl --npy-dir gadsO
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import argparse, json, sys, time
import multiprocessing as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from outline import climb
from slp import slp


def job(args):
    w, h, f0, seed, steps, npy_dir = args
    t0 = time.time()
    n, S, _ = climb((w, h, f0, seed, steps, None))
    d = n - w * h
    r = dict(w=w, h=h, f0=f0, seed=seed, n=int(n), d=int(d), f=None, dt=0.0)
    if S is not None and d >= 1:
        x, y, t, L, v, L0 = slp(S[:, 0], S[:, 1], S[:, 2], iters=400, ex=w, ey=h)
        if v < 1e-10:
            r['f'] = float(L)
            p = os.path.join(npy_dir, f'o_{w}_{h}_{d}_{L:.6f}_{seed}_{f0:.2f}.npy')
            np.save(p, np.c_[x, y, t])
    r['dt'] = time.time() - t0
    return r


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--w', type=int, nargs=2, default=[8, 24]); ap.add_argument('--h', type=int, nargs=2, default=[4, 14])
    ap.add_argument('--f0', type=float, nargs='+', default=[0.72, 0.78, 0.84])
    ap.add_argument('--seeds', type=int, default=2); ap.add_argument('--seed0', type=int, default=0)
    ap.add_argument('--steps', type=int, default=2000); ap.add_argument('--procs', type=int, default=4)
    ap.add_argument('--out', default='outline_sweep.jsonl'); ap.add_argument('--npy-dir', default='gadsO')
    a = ap.parse_args()
    os.makedirs(a.npy_dir, exist_ok=True)
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            r = json.loads(line); done.add((r['w'], r['h'], round(r['f0'], 4), r['seed']))
    jobs = [(w, h, f0, a.seed0 + s, a.steps, a.npy_dir)
            for w in range(a.w[0], a.w[1] + 1) for h in range(a.h[0], a.h[1] + 1) if h <= w
            for f0 in a.f0 for s in range(a.seeds)
            if (w, h, round(f0, 4), a.seed0 + s) not in done]
    # big boxes first so the tail is short
    jobs.sort(key=lambda j: -j[0] * j[1])
    print(f'{len(jobs)} jobs', flush=True)
    with mp.get_context('spawn').Pool(a.procs) as pool, open(a.out, 'a') as fo:
        for k, r in enumerate(pool.imap_unordered(job, jobs)):
            fo.write(json.dumps(r) + '\n'); fo.flush()
            if r['f'] is not None:
                print(f"[{k + 1}/{len(jobs)}] {r['w']}x{r['h']}+{r['d']} f {r['f']:.6f} (f0 {r['f0']}, {r['dt']:.0f}s)", flush=True)
