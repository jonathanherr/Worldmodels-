"""GA over band-end gadgets with insert moves, screened by a learned critic.

The rescan says the closest small-m designs (k^2+1, k^2+2 at k 20-31) are 1-2 squares short, while a
pure squeeze moves the end face by at most ~0.01 (a whole column needs ~1). So the lever is inserting a
square into an end at (almost) the same face: waste change de = m (c' - c) - 1 < 0 is a gain.

Each generation, every parent proposes many candidate moves (insert into one of its largest holes, at
angle 0 or -phi, or relocate one square into a hole). A critic (gradient boosting on hole features,
trained online on every exact evaluation so far) picks the few that get the exact float64
relax + squeeze (endgadget.Gadget). Children with lower waste join the population. Every evaluation is
logged with its features: that log is the critic's training corpus.

usage: python -m research.pretrain.ga_ends SYNTH_DIR OUT minutes [r,f r,f ...]
"""
import json, math, os, sys, time, random
from multiprocessing import Pool
import numpy as np
from research.corner_scale.seqend import Seq
from research.corner_scale.fastdp import dp as fdp
from research.corner_scale.dpbuild import squares
from research.corner_scale.endgadget import Gadget

N_FEAT = 12


def holes(x, y, H, c, cs, sn, rng, K=3000, top=8):
    yy = rng.uniform(0.5, H - 0.5, K)
    xmax = (c - 0.6 + yy * sn) / cs
    xx = 0.5 + rng.uniform(0, 1, K) * np.maximum(xmax - 0.5, 0.05)
    d = np.sqrt((xx[:, None] - x[None]) ** 2 + (yy[:, None] - y[None]) ** 2).min(1)
    order = np.argsort(-d); out = []
    for i in order:
        if all((xx[i] - a) ** 2 + (yy[i] - b) ** 2 > 0.25 for a, b, _ in out):
            out.append((xx[i], yy[i], d[i]))
        if len(out) >= top: break
    return out


def features(st, hole, ang_tilted, move, slack):
    x, y, r, f, c, cs, sn = st['x'], st['y'], st['r'], st['f'], st['c'], st['cs'], st['sn']
    H = r + f
    hx, hy, hd = hole
    near = np.sqrt((x - hx) ** 2 + (y - hy) ** 2)
    u = hx * cs - hy * sn
    return [r, f, hd, (c - u), hy / H, min(hy, H - hy), hx, float(ang_tilted), float(move), slack,
            st['e'] / H, (near < 1.6).sum()]


def evaluate(job):
    st, hole, ang_tilted, move, slack, seed = job
    rng = np.random.default_rng(seed)
    r, f, phi = st['r'], st['f'], st['phi']
    G = Gadget(r, f, phi)
    x, y, t = np.array(st['x']), np.array(st['y']), np.array(st['t'])
    hx, hy, _ = hole
    a = -phi if ang_tilted else 0.0
    if move == 0:   # insert
        x = np.append(x, hx); y = np.append(y, hy); t = np.append(t, a)
    else:           # relocate the square nearest a random point to the hole
        k = int(rng.integers(len(x))); x[k], y[k], t[k] = hx, hy, a
    t0 = time.time()
    res = G.squeeze(x, y, t, st['c'] + slack, rounds=24)
    out = {'ok': False, 'secs': 0.0}
    if res is not None:
        nx, ny, nt, nc = res
        if G.feasible(nx, ny, nt, nc):
            out = {'ok': True, 'x': nx.tolist(), 'y': ny.tolist(), 't': nt.tolist(), 'c': float(nc),
                   'N': len(nx), 'e': float(G.waste(len(nx), nc))}
    out['secs'] = time.time() - t0
    return out


def dp_parents(synth, r, f, n_a0):
    z = np.load(os.path.join(synth, f'dp_r{r}.npz'))
    rows, jm, ff, ph = z['rows'], z['jmax'], z['f'], z['phi']
    i = int(np.argmin(abs(ff - f))); Jmax = int(jm[i]); phi = float(ph[i])
    sel = rows[np.isclose(rows[:, 0], ff[i])]
    best = {}
    for row in sel:
        a0 = round(row[1], 9)
        if a0 not in best or row[6] < best[a0][6]: best[a0] = row
    picks = sorted(best.values(), key=lambda row: row[6])[:n_a0]
    out = []
    for row in picks:
        S = Seq(r, float(ff[i]), row[1], row[2], Jmax)
        J, N = fdp(S, 'dec' if row[3] == 0 else 'inc', want_seq=True)
        sq = squares(S, J)
        out.append({'r': r, 'f': float(ff[i]), 'phi': S.phi, 'cs': S.cs, 'sn': S.sn, 'a0': float(row[1]),
                    'x': sq[:, 0].tolist(), 'y': sq[:, 1].tolist(), 't': sq[:, 2].tolist(), 'c': float(S.c),
                    'c_dp': float(S.c), 'N': len(sq), 'N_dp': int(N), 'e': float(S.waste(N)), 'gen': 0, 'path': []})
    return out


NEED = {  # (r, f): {k: m needed to beat our side at n = k^2 + m}
    (8, 0.52): {25: 1, 26: 1, 27: 1, 28: 1, 29: 1, 30: 1, 31: 1},
    (11, 0.535): {31: 2},
    (8, 0.535): {20: 1, 21: 1, 22: 1, 23: 1, 24: 1, 29: 2, 30: 2},
}


def table_of(synth, r, f):
    z = np.load(os.path.join(synth, f'dp_r{r}.npz'))
    rows, ff, ph = z['rows'], z['f'], z['phi']
    i = int(np.argmin(abs(ff - f))); sel = rows[np.isclose(rows[:, 0], ff[i])]
    best = {}
    for row in sel:
        a0 = round(row[1], 9)
        if a0 not in best or row[4] > best[a0][0]: best[a0] = (row[4], row[5])
    return np.array([v[0] for v in best.values()]), np.array([v[1] for v in best.values()]), float(ph[i])


def shortfall(N, c, phi, r, f, need):
    out = {}
    for k, m in need.items():
        D = (k + f) * math.cos(phi) - (r + f) * math.sin(phi)
        M = np.floor(D - c[:, None] - c[None, :] + 1e-12)
        cnt = np.where(M >= 0, N[:, None] + N[None, :] + M * (r + 1), -1)
        out[k] = int(m - (cnt.max() + (k - r) * k - k * k))
    return out


def near_pairs(N, c, phi, r, f, need, slack=2):
    used = set()
    for k, m in need.items():
        D = (k + f) * math.cos(phi) - (r + f) * math.sin(phi)
        M = np.floor(D - c[:, None] - c[None, :] + 1e-12)
        cnt = np.where(M >= 0, N[:, None] + N[None, :] + M * (r + 1), -1) + (k - r) * k - k * k
        for i, j in zip(*np.where(cnt >= m - slack)):
            used.add(round(float(c[i]), 6)); used.add(round(float(c[j]), 6))
    return used


def main():
    synth, outdir, minutes = sys.argv[1], sys.argv[2], float(sys.argv[3])
    targets = [(int(a), float(b)) for a, b in (s.split(',') for s in sys.argv[4:])] or [(8, 0.52), (11, 0.535), (8, 0.535)]
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(0)
    pop = []
    base = {}
    for r, f in targets:
        N0, c0, ph0 = table_of(synth, r, f); base[(r, f)] = (N0, c0, ph0)
        need = NEED.get((r, f), {})
        used = near_pairs(N0, c0, ph0, r, f, need)
        cand = dp_parents(synth, r, f, 80)
        sel = [p for p in cand if round(p['c'], 6) in used][:16] or cand[:8]
        pop += sel
        print('target', r, f, 'shortfall', shortfall(N0, c0, ph0, r, f, need), 'parents', len(sel), flush=True)
    for p in pop:
        p['x'], p['y'] = np.array(p['x']), np.array(p['y'])
    log = open(os.path.join(outdir, 'evals.jsonl'), 'a')
    best_file = os.path.join(outdir, 'best_ends.jsonl')
    X, Y = [], []
    critic = None
    t_start = time.time(); gen = 0
    with Pool(4) as pool:
        while time.time() - t_start < minutes * 60:
            gen += 1
            jobs, feats = [], []
            for p in pop:
                H = p['r'] + p['f']
                cands = []
                for hole in holes(p['x'], p['y'], H, p['c'], p['cs'], p['sn'], rng):
                    for ang in (0, 1):
                        for move in (0, 1):
                            for slack in (0.0, 0.05):
                                cands.append((hole, ang, move, slack))
                F = np.array([features(p, *cd) for cd in cands])
                if critic is not None:
                    score = critic.predict(F) + rng.normal(0, 0.05, len(F))  # predicted de, small exploration noise
                    pick = np.argsort(score)[:2]
                else:
                    pick = rng.choice(len(cands), 2, replace=False)
                for i in pick:
                    st = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in p.items()}
                    jobs.append((st, *cands[i], int(rng.integers(1 << 30)))); feats.append((p, F[i], cands[i]))
            results = pool.map(evaluate, jobs)
            children = []
            for (p, fv, cd), res in zip(feats, results):
                de = (res['e'] - p['e']) if res['ok'] else 2.0
                X.append(fv); Y.append(min(de, 2.0))
                log.write(json.dumps({'gen': gen, 'r': p['r'], 'f': p['f'], 'a0': p['a0'], 'feat': list(map(float, fv)),
                                      'ok': res['ok'], 'de': de, 'parent_e': p['e'], 'secs': res['secs'],
                                      'critic': critic is not None}) + '\n')
                if res['ok'] and de < -1e-4:
                    ch = dict(p); ch.update(x=np.array(res['x']), y=np.array(res['y']), t=res['t'], c=res['c'],
                                            N=res['N'], e=res['e'], gen=gen, path=p['path'] + [list(map(float, cd[0])) + list(cd[1:])])
                    children.append(ch)
                    with open(best_file, 'a') as bf:
                        bf.write(json.dumps({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in ch.items()}) + '\n')
            log.flush()
            pop += children
            allch = getattr(main, 'allch', []); allch += children; main.allch = allch
            # keep the best 8 per (r, f, a0-rounded) lineage cap: best 24 overall per target
            keep = []
            for r, f in targets:
                grp = sorted([p for p in pop if p['r'] == r and abs(p['f'] - f) < 1e-6], key=lambda p: p['e'])
                seen = {}
                for q in grp:
                    b = round(q['a0'], 3)
                    if seen.get(b, 0) < 2: keep.append(q); seen[b] = seen.get(b, 0) + 1
            pop = keep
            if len(Y) >= 40:
                from sklearn.ensemble import GradientBoostingRegressor
                critic = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05).fit(np.array(X), np.array(Y))
            okr = np.mean([r_['ok'] for r_ in results]); gain = sum(1 for c_ in children)
            sf = []
            for r, f in targets:
                N0, c0, ph0 = base[(r, f)]
                ch = [q for q in allch if q['r'] == r and abs(q['f'] - f) < 1e-6]
                N1 = np.append(N0, [q['N'] for q in ch]); c1 = np.append(c0, [q['c'] for q in ch])
                sf.append(f"r{r} f{f} short {shortfall(N1, c1, ph0, r, f, NEED.get((r, f), {}))}")
            print('   ', ' | '.join(sf), flush=True)
            summary = ', '.join(f"r{r} f{f}: {min(p['e'] for p in pop if p['r'] == r and abs(p['f'] - f) < 1e-6) / (r + f):.4f}" for r, f in targets)
            print(f'gen {gen} t={time.time() - t_start:.0f}s evals {len(jobs)} feasible {okr:.2f} gains {gain} best/row {summary}', flush=True)


if __name__ == '__main__':
    main()
