"""Score ends by the squares they allow: for every n = k^2 + m (k <= 31) find the smallest tabulated f
whose best one-band design (two ends + full columns + axis rows) holds n, using DP ends and, if given,
squeezed ends (same N, smaller face c). Compares with our side and the per-n leader.

usage: python -m research.pretrain.rescan SYNTH_DIR [squeezed.jsonl]
"""
import sys, os, json, math, csv, glob
import numpy as np


def tables(d, squeezed=None):
    T = {}
    for p in sorted(glob.glob(os.path.join(d, 'dp_r*.npz'))):
        r = int(os.path.basename(p)[4:-4]); z = np.load(p)
        rows, jm, ff, ph = z['rows'], z['jmax'], z['f'], z['phi']
        for f, J, phi in zip(ff, jm, ph):
            sel = rows[np.isclose(rows[:, 0], f)]
            ends = {}
            for row in sel:  # per a0: best N (c depends only on a0)
                a0 = round(row[1], 9)
                if a0 not in ends or row[4] > ends[a0][0]:
                    ends[a0] = (row[4], row[5])
            N = np.array([v[0] for v in ends.values()]); c = np.array([v[1] for v in ends.values()])
            T[(r, round(float(f), 4))] = {'N': N, 'c': c, 'phi': float(phi)}
    if squeezed:
        for line in open(squeezed):
            o = json.loads(line)
            if not o.get('ok'): continue
            key = (o['r'], round(o['f'], 4))
            if key in T:  # add the squeezed end as an extra entry
                T[key]['N'] = np.append(T[key]['N'], o['N_dp']); T[key]['c'] = np.append(T[key]['c'], o['c_sq'])
    return T


def band_count(t, r, f, Lb):
    m = r + 1; H = r + f
    D = Lb * math.cos(t['phi']) - H * math.sin(t['phi'])
    M = np.floor(D - t['c'][:, None] - t['c'][None, :] + 1e-12)
    cnt = np.where(M >= 0, t['N'][:, None] + t['N'][None, :] + M * m, -1)
    return int(cnt.max())


def main():
    T = tables(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
    fs = sorted(set(f for _, f in T)); rs = sorted(set(r for r, _ in T))
    R = {int(x['n']): x for x in csv.DictReader(open('/mnt/project-files/squares/leaderboard-20261103/gap_vs_best_per_n.csv'))}
    # best m reachable per (k, f) with one band
    cap = {}
    for k in range(10, 32):
        for f in fs:
            best = -10**9
            for r in rs:
                if r >= k or (r, f) not in T: continue
                best = max(best, band_count(T[(r, f)], r, f, k + f) + (k - r) * k - k * k)
            cap[(k, f)] = best
    out = []
    win_o = win_l = 0; gain = 0.0
    for n in range(100, 1001):
        k = math.isqrt(n); m = n - k * k
        if m == 0 or k > 31: continue
        f = next((f for f in fs if cap[(k, f)] >= m), None)
        if f is None: continue
        ours = float(R[n]['our_L']) - k; lead = float(R[n]['best_L']) - k
        out.append((n, k, m, f, round(ours, 5), round(lead, 5)))
        if f < ours - 1e-9: win_o += 1; gain += (ours - f) / math.sqrt(n)
        if f < lead - 1e-9: win_l += 1
    gap = [o[3] - o[4] for o in out]
    print('n covered', len(out), 'beat ours', win_o, 'score gain', round(gain, 5), 'beat leader', win_l)
    print('median band f minus our f', round(float(np.median(gap)), 4), 'best', round(min(gap), 4))
    best = sorted(out, key=lambda o: o[3] - o[4])[:10]
    for o in best: print(o)
    return out


if __name__ == '__main__':
    main()
