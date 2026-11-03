"""Synthetic band-end corpus: exact DP ends over (r, f, a0, off, kind), plus CPU-squeezed versions of
the best few per (r, f).

Stage 'dp'      -> dp_r{r}.npz: every (f, a0, off, kind) with DP count N, face c, waste e.
Stage 'squeeze' -> squeeze_r{r}_f{f}.json: top-K distinct-a0 DP ends relaxed and squeezed with
                   endgadget.Gadget (float64, exact feasibility check), new face c' and waste e'.

usage: python -m research.pretrain.synth_ends dp OUT r0 r1 f0 f1 df
       python -m research.pretrain.synth_ends squeeze OUT r0 r1 f0 f1 df K
"""
import json, math, os, sys, time
from multiprocessing import Pool
import numpy as np
from research.corner_scale.seqend import Seq
from research.corner_scale.fastdp import dp as fdp
from research.corner_scale.dpbuild import squares
from research.corner_scale.endgadget import Gadget
from research.corner_scale.dpend import phi_for

NA, NOFF = 80, 8


def dp_rf(args):
    r, f = args
    try:
        phi = phi_for(r, f)
    except Exception:
        return None
    hs = (r + f) * math.sin(phi)
    Jmax = int(math.ceil(hs)) + 6
    rows = []
    for a0 in np.linspace(-hs - 1.5, -hs - 0.5, NA, endpoint=False):
        for off in np.linspace(0, f, NOFF):
            S = Seq(r, f, a0, off, Jmax)
            for kind in (0, 1):
                _, N = fdp(S, 'dec' if kind == 0 else 'inc')
                rows.append((f, a0, off, kind, N, S.c, S.waste(N)))
    return np.array(rows), Jmax, phi


def squeeze_one(args):
    r, f, a0, off, kind, Jmax = args
    S = Seq(r, f, a0, off, Jmax)
    J, N = fdp(S, 'dec' if kind == 0 else 'inc', want_seq=True)
    sq = squares(S, J)
    G = Gadget(r, f, S.phi)
    t0 = time.time()
    res = G.squeeze(sq[:, 0].copy(), sq[:, 1].copy(), sq[:, 2].copy(), S.c + 1e-3)
    out = {'r': r, 'f': f, 'a0': a0, 'off': off, 'kind': kind, 'Jmax': Jmax, 'N_dp': int(N), 'n_sq': len(sq),
           'c_dp': S.c, 'e_dp': S.waste(N), 'J': [int(v) for v in J], 'secs': None}
    if res is None:
        out['ok'] = False
    else:
        x, y, t, c = res
        ok = G.feasible(x, y, t, c)
        out.update(ok=bool(ok), c_sq=float(c), e_sq=float(G.waste(len(x), c)),
                   x=x.tolist(), y=y.tolist(), t=t.tolist())
    out['secs'] = time.time() - t0
    return out


def main():
    stage, out = sys.argv[1], sys.argv[2]
    r0, r1 = int(sys.argv[3]), int(sys.argv[4])
    fs = np.round(np.arange(float(sys.argv[5]), float(sys.argv[6]) + 1e-9, float(sys.argv[7])), 4)
    os.makedirs(out, exist_ok=True)
    with Pool(4) as pool:
        if stage == 'dp':
            for r in range(r0, r1 + 1):
                p = os.path.join(out, f'dp_r{r}.npz')
                if os.path.exists(p): continue
                res = pool.map(dp_rf, [(r, float(f)) for f in fs])
                good = [x for x in res if x is not None]
                rows = np.concatenate([g[0] for g in good])
                np.savez(p, rows=rows, jmax=np.array([g[1] for g in good]), phi=np.array([g[2] for g in good]),
                         f=np.array([g[0][0, 0] for g in good]))
                print('r', r, rows.shape, flush=True)
        else:
            K = int(sys.argv[8])
            jobs = []
            for r in range(r0, r1 + 1):
                d = np.load(os.path.join(out, f'dp_r{r}.npz'))
                rows, jm, ff = d['rows'], d['jmax'], d['f']
                for f in fs:
                    sel = rows[np.isclose(rows[:, 0], f)]
                    if not len(sel): continue
                    Jmax = int(jm[np.argmin(abs(ff - f))])
                    sel = sel[np.argsort(sel[:, 6])]
                    seen = []
                    for row in sel:
                        if any(abs(row[1] - s) < 0.05 for s in seen): continue
                        seen.append(row[1])
                        jobs.append((r, float(f), float(row[1]), float(row[2]), int(row[3]), Jmax))
                        if len(seen) >= K: break
            print('squeeze jobs', len(jobs), flush=True)
            with open(os.path.join(out, 'squeezed.jsonl'), 'a') as fh:
                for i, o in enumerate(pool.imap_unordered(squeeze_one, jobs)):
                    fh.write(json.dumps(o) + '\n'); fh.flush()
                    if i % 50 == 0:
                        print(i, o['r'], o['f'], o.get('e_dp'), o.get('e_sq'), round(o['secs'], 1), flush=True)


if __name__ == '__main__':
    main()
