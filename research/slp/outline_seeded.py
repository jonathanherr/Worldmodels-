"""Outline climb seeded from existing gadgets: try to fit one more square (d+1) at the gadget's own f,
or the same count at a smaller f, by discrete changes of the tilted block's outline + exact frame refill.

    python research/slp/outline_seeded.py gads_dir/*.npy --procs 31 --out seeded.jsonl --npy-dir gadsS
Gadget file names end in _{w}_{h}_{d}[_...].npy (farm style n{n}_{c}_{w}_{h}_{d}.npy).
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import argparse, json, math, re, sys, time
import multiprocessing as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import outline
from outline import climb, seed_from, evaluate, tilted_squares, pair_overlap
from slp import slp, side


def parse(p):
    b = os.path.basename(p)[:-4]
    m = re.match(r'n\d+_\d+_(\d+)_(\d+)_(\d+)$', b) or re.search(r'_(\d+)_(\d+)_(\d+)(?:_[^_]*)*$', b)
    return tuple(int(v) for v in m.groups())


def job(args):
    p, steps, npy_dir, seed = args
    w, h, d = parse(p)
    q = np.load(p)
    f_g, _, _ = side(q[:, 0], q[:, 1], q[:, 2], w, h)
    ang = np.mod(q[:, 2], math.pi / 2)
    ntilt = int(((ang > 0.05) & (ang < math.pi / 2 - 0.05)).sum())
    r = dict(src=os.path.basename(p), w=w, h=h, d=d, f=float(f_g), ntilt=ntilt, out=[])
    if ntilt < 6:
        return r
    try:
        st = seed_from(q, w, h)
    except Exception as e:
        r['err'] = str(e); return r
    # the seed's own squares that are not on the fitted block are kept as candidates
    T = tilted_squares(*st)
    dd_ = np.hypot(q[:, None, 0] - T[None, :, 0], q[:, None, 1] - T[None, :, 1]).min(1)
    outline.EXTRA = q[dd_ > 0.05].copy(); outline.extra_f[0] = float(f_g)
    n_seed, _ = evaluate(st, w, h, f_g + 1e-9)
    r['seed_d'] = int(n_seed - w * h)
    for f0 in (f_g + 0.004, f_g - 0.01):
        n, S, _ = climb((w, h, f0, seed, steps, st))
        dd = n - w * h
        o = dict(f0=float(f0), d=int(dd), f=None)
        if S is not None and (dd > d or (dd == d and f0 < f_g)):
            x, y, t, L, v, L0 = slp(S[:, 0], S[:, 1], S[:, 2], iters=400, ex=w, ey=h)
            if v < 1e-10:
                o['f'] = float(L)
                if dd > d or L < f_g - 1e-6:
                    np.save(os.path.join(npy_dir, f's_{w}_{h}_{dd}_{L:.6f}_{os.path.basename(p)}'), np.c_[x, y, t])
        r['out'].append(o)
    return r


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('files', nargs='+'); ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--procs', type=int, default=4); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='seeded.jsonl'); ap.add_argument('--npy-dir', default='gadsS')
    a = ap.parse_args()
    os.makedirs(a.npy_dir, exist_ok=True)
    jobs = [(p, a.steps, a.npy_dir, a.seed) for p in a.files]
    with mp.get_context('spawn').Pool(a.procs) as pool, open(a.out, 'a') as fo:
        for k, r in enumerate(pool.imap_unordered(job, jobs)):
            fo.write(json.dumps(r) + '\n'); fo.flush()
            for o in r['out']:
                if o['f'] is not None and (o['d'] > r['d'] or o['f'] < r['f'] - 1e-6):
                    print(f"[{k + 1}/{len(jobs)}] {r['src']} {r['w']}x{r['h']}+{r['d']} f {r['f']:.6f} -> +{o['d']} f {o['f']:.6f}", flush=True)
