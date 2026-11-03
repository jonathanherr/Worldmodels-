"""Scan n = k^2 + m for band constructions (one band, or horizontal + vertical band) that beat
our current side, using tabulated end gadgets."""
import sys, json, math, csv, glob
import numpy as np

def load_tables(pattern):
    T = {}
    for p in glob.glob(pattern):
        for key, v in json.load(open(p)).items():
            if v is None: continue
            r, f = key.split(','); T[(int(r), round(float(f), 4))] = v
    for v in T.values():
        v['a0'] = np.array(v['a0']); v['N'] = np.array(v['N'])
        v['c'] = v['a0'] + v['Jmax']
    return T

def band_count(t, r, f, Lb):
    m = r + 1; H = r + f
    cs, sn = math.cos(t['phi']), math.sin(t['phi'])
    D = Lb * cs - H * sn
    M = np.floor(D - t['c'][:, None] - t['c'][None, :] + 1e-12)
    cnt = t['N'][:, None] + t['N'][None, :] + M * m
    cnt = np.where(M >= 0, cnt, -1)
    i, j = np.unravel_index(np.argmax(cnt), cnt.shape)
    return int(cnt[i, j]), (int(i), int(j), int(M[i, j]))

def best_for(T, k, m, fs, rs):
    n = k * k + m
    for f in fs:
        for r1 in rs:
            if r1 >= k or (r1, f) not in T: continue
            c1, a1 = band_count(T[(r1, f)], r1, f, k + f)
            if c1 + (k - r1) * k >= n:
                return f, ('one', r1, a1)
            for r2 in rs:
                if r2 >= k - r1 or (r2, f) not in T: continue
                c2, a2 = band_count(T[(r2, f)], r2, f, k - r1)
                if c1 + c2 + (k - r1) * (k - r2) >= n:
                    return f, ('two', r1, a1, r2, a2)
    return None

if __name__ == '__main__':
    T = load_tables(sys.argv[1])
    fs = sorted(set(f for _, f in T))
    rs = sorted(set(r for r, _ in T))
    R = {int(x['n']): x for x in csv.DictReader(open('/mnt/project-files/squares/leaderboard-20261103/gap_vs_best_per_n.csv'))}
    wins = []; tot = 0
    for n in range(100, 1001):
        k = math.isqrt(n); m = n - k * k
        if m == 0: continue
        ours = float(R[n]['our_L']) - k; lead = float(R[n]['best_L']) - k
        cand = [f for f in fs if f < ours - 1e-9]
        b = best_for(T, k, m, cand, rs)
        if b:
            g = (ours - b[0]) / math.sqrt(n); tot += g
            wins.append((n, k, m, round(ours, 4), b[0], round(lead, 4), b[1]))
    for w in wins: print(w)
    print('wins', len(wins), 'score gain (grid f)', round(tot, 5), 'f range', fs[0], fs[-1], 'rs', rs[0], rs[-1])
