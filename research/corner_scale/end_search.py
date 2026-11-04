import sys, math, json, time
import numpy as np
sys.path.insert(0, '/home/claude/worldmodels-')
from research.corner_scale.end_from_greedy import greedy_gadget

def run(r, f, X0, iters, seed, out, init=None):
    rng = np.random.default_rng(seed)
    if init:
        from research.corner_scale.endgadget import Gadget
        g = json.load(open(init)); G = Gadget(g['r'], g['f'], g['phi'])
        sq = np.stack([g['x'], g['y'], g['t']], 1); c = g['c']
    else:
        G, sq, c = greedy_gadget(r, f, X0)
    x, y, t = sq[:, 0].copy(), sq[:, 1].copy(), sq[:, 2].copy()
    cx, cy, ct, cc = G.compact(x, y, t, c)
    best = G.squeeze(cx, cy, ct, cc + 0.002)
    bw = G.waste(len(best[0]), best[3])
    print(f'start N={len(best[0])} waste={bw:.4f} per row {bw/G.H:.4f}', flush=True)
    t0 = time.time()
    for it in range(iters):
        x, y, t, c = [a.copy() if hasattr(a, 'copy') else a for a in best]
        mv = rng.random()
        if mv < 0.5:  # insert
            # random point inside trapezoid
            yy = rng.uniform(0.5, G.H - 0.5)
            xmax = (c + yy * G.sn) / G.cs
            xx = rng.uniform(0.5, max(0.6, xmax - 0.5))
            tt = 0.0 if rng.random() < 0.5 else -G.phi
            x, y, t = np.append(x, xx), np.append(y, yy), np.append(t, tt)
            c = c + 1.0 / G.m * rng.uniform(0.2, 1.0)
        elif mv < 0.75:  # kick
            s = rng.choice([0.03, 0.1, 0.3])
            x = x + rng.normal(0, s, len(x)); y = y + rng.normal(0, s, len(y)); t = t + rng.normal(0, s / 3, len(t))
        else:  # remove one and insert one elsewhere
            i = rng.integers(len(x))
            x, y, t = np.delete(x, i), np.delete(y, i), np.delete(t, i)
            yy = rng.uniform(0.5, G.H - 0.5); xmax = (c + yy * G.sn) / G.cs
            xx = rng.uniform(0.5, max(0.6, xmax - 0.5)); tt = 0.0 if rng.random() < 0.5 else -G.phi
            x, y, t = np.append(x, xx), np.append(y, yy), np.append(t, tt)
        cx, cy, ct, cc = G.compact(x, y, t, c)
        res = G.squeeze(cx, cy, ct, cc + 0.002)
        if res is None:
            continue
        w = G.waste(len(res[0]), res[3])
        if w < bw - 1e-6:
            best, bw = res, w
            print(f'it {it} mv {mv:.2f} N={len(res[0])} waste={bw:.4f} per row {bw/G.H:.4f} ({time.time()-t0:.0f}s)', flush=True)
            json.dump({'r': r, 'f': f, 'phi': G.phi, 'N': len(res[0]), 'c': res[3], 'waste': bw,
                       'x': res[0].tolist(), 'y': res[1].tolist(), 't': res[2].tolist()}, open(out, 'w'))
    print('done', bw, bw / G.H)

if __name__ == '__main__':
    r, f, X0, iters, seed = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    run(r, f, X0, iters, seed, sys.argv[6], sys.argv[7] if len(sys.argv) > 7 else None)
