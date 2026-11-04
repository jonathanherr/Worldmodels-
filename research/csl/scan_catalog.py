"""Score every CSL catalog band (exact f, exact end fronts) on all n = k^2+m, k 10..31, one band or
horizontal + vertical band (second band reuses the same f and any catalog r at that f)."""
import sys, json, math, csv
from multiprocessing import Pool
sys.path.insert(0, '.')
from research.csl.catalog import catalog
from research.csl.exact_end import exact_table, band_max

def work(c):
    try:
        t = exact_table(c['r'], c['f'])
    except Exception as e:
        return c, None
    return c, t

if __name__ == '__main__':
    import os
    os.environ['OMP_NUM_THREADS'] = '1'
    cat = catalog(max_sigma=int(sys.argv[1]) if len(sys.argv) > 1 else 100)
    with Pool(4) as P:
        res = P.map(work, cat, chunksize=1)
    R = {int(x['n']): x for x in csv.DictReader(open('/mnt/project-files/squares/leaderboard-20261103/gap_vs_best_per_n.csv'))}
    tabs = [(c, t) for c, t in res if t]
    json.dump([(c, {'phi': t['phi'], 'Jmax': t['Jmax'], 'front': t['front']}) for c, t in tabs], open(sys.argv[2], 'w'))
    best = {}  # n -> (f, design)
    near = []
    for c, t in tabs:
        r, f = c['r'], c['f']
        for k in range(r + 1, 32):
            cnt, how = band_max(t, r, f, k + f)
            m1 = cnt + (k - r) * k - k * k
            designs = [(m1, ('one', r, c['p'], c['q']))]
            # vertical band in the remaining (k - r) rows, same f, same angle family
            for c2, t2 in tabs:
                if abs(c2['f'] - f) > 1e-12 or c2['r'] >= k - r: continue
                cnt2, _ = band_max(t2, c2['r'], f, k - r)
                designs.append((cnt + cnt2 + (k - r) * (k - c2['r']) - k * k, ('two', r, c2['r'], c['p'], c['q'])))
            mm, d = max(designs)
            for m in range(1, 2 * k + 1):
                n = k * k + m
                if n > 1000: break
                ours = float(R[n]['our_L']) - k; lead = float(R[n]['best_L']) - k
                if f < ours - 1e-9:
                    if mm >= m:
                        if n not in best or f < best[n][0]:
                            best[n] = (f, d, ours, lead)
                    elif mm >= m - 2:
                        near.append((n, k, m, m - mm, round(f, 5), round(ours, 5), round(lead, 5), d))
    tot = 0
    for n in sorted(best):
        f, d, ours, lead = best[n]; tot += (ours - f) / math.sqrt(n)
        print('WIN', n, round(f, 6), 'ours', round(ours, 6), 'lead', round(lead, 6), d)
    print('wins', len(best), 'gain', round(tot, 6))
    near.sort(key=lambda x: (x[3], x[0]))
    print('near misses (squares short):', len(near))
    for x in near[:40]: print('  ', x)
