"""For each k and tabulated f: the largest m reachable by one band or horizontal + vertical band.
Then min f per n = k^2 + m, compared with our side."""
import sys, json, math, csv
from functools import lru_cache
import numpy as np
from research.corner_scale.scan import load_tables, band_count

T = load_tables(sys.argv[1])
fs = sorted(set(f for _, f in T)); rs = sorted(set(r for r, _ in T))

@lru_cache(maxsize=None)
def bc(r, f, Lb):
    t = T.get((r, f))
    if t is None: return -10**9, None
    return band_count(t, r, f, Lb)

R = {int(x['n']): x for x in csv.DictReader(open('/mnt/project-files/squares/leaderboard-20261103/gap_vs_best_per_n.csv'))}
res = {}
for k in range(10, 32):
    for f in fs:
        best = (-10**9, None)
        for r1 in rs:
            if r1 >= k: continue
            c1, a1 = bc(r1, f, round(k + f, 6))
            if c1 < 0: continue
            v = c1 + (k - r1) * k - k * k
            if v > best[0]: best = (v, ('one', r1, a1))
            for r2 in rs:
                if r2 >= k - r1: continue
                c2, a2 = bc(r2, f, k - r1)
                if c2 < 0: continue
                v = c1 + c2 + (k - r1) * (k - r2) - k * k
                if v > best[0]: best = (v, ('two', r1, a1, r2, a2))
        res[(k, f)] = best
wins = []; tot = 0
for n in range(100, 1001):
    k = math.isqrt(n); m = n - k * k
    if m == 0 or k > 31: continue
    ours = float(R[n]['our_L']) - k; lead = float(R[n]['best_L']) - k
    for f in fs:
        if f >= ours - 1e-9: break
        if res[(k, f)][0] >= m:
            g = (ours - f) / math.sqrt(n); tot += g
            wins.append((n, k, m, round(ours, 4), f, round(lead, 4), res[(k, f)][1]))
            break
for w in wins: print(w)
print('wins', len(wins), 'score gain (grid f)', round(tot, 5), 'f range', fs[0], fs[-1], 'rs', rs[0], rs[-1])
json.dump([list(map(str, w)) for w in wins], open(sys.argv[2], 'w'))
if len(sys.argv) > 3:
    for k in [int(x) for x in sys.argv[3].split(',')]:
        print(k, [(f, res[(k, f)][0], res[(k, f)][1][:2] if res[(k, f)][1] else None) for f in fs if abs(f * 1000 % 20) < 1e-6])
