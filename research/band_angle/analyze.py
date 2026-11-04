"""Smallest f per n = k^2 + m reachable by one band, tight angle vs free angle; compare with ours/leader."""
import sys, json, math, csv
from collections import defaultdict

rows = [json.loads(l) for p in sys.argv[1:] for l in open(p)]
R = {int(x['n']): x for x in csv.DictReader(open('/mnt/project-files/squares/leaderboard-20261103/gap_vs_best_per_n.csv'))}
tight = defaultdict(lambda: -10**9); free = defaultdict(lambda: -10**9); arg = {}
for o in rows:
    for k, c in o['cnt'].items():
        key = (int(k), o['f']); m = c - int(k) ** 2
        if o['dphi'] == 0: tight[key] = max(tight[key], m)
        if m > free[key]: free[key] = m; arg[key] = (o['r'], o['dphi'])
fs = sorted({o['f'] for o in rows})
gain_n = 0; better_than_ours = []
for n in range(226, 1001):
    k = math.isqrt(n); m = n - k * k
    if m == 0 or k > 31: continue
    ft = next((f for f in fs if tight[(k, f)] >= m), None)
    ff = next((f for f in fs if free[(k, f)] >= m), None)
    if ff is None: continue
    ours = float(R[n]['our_L']) - k; lead = float(R[n]['best_L']) - k
    if ft is None or ff < ft - 1e-9:
        gain_n += 1
        print('angle helps', n, k, m, 'tight f', ft, 'free f', ff, arg[(k, ff)], 'ours', round(ours, 4), 'lead', round(lead, 4))
    if ff < ours - 1e-9: better_than_ours.append((n, ff, ours))
print('n where free angle lowers f:', gain_n, ' beat ours:', better_than_ours)
# shortfall at our f for k^2+1
for k in range(15, 32):
    n = k * k + 1; ours = float(R[n]['our_L']) - k
    fb = max([f for f in fs if f < ours - 1e-9], default=None)
    if fb: print('k', k, 'at f', fb, 'tight m', tight[(k, fb)], 'free m', free[(k, fb)], 'best dphi', arg.get((k, fb)))
