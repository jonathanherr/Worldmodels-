import math, csv, sys
from scipy.optimize import brentq
def s(r, f):
    try:
        p = brentq(lambda p: (r + 1) * math.cos(p) + math.sin(p) - (r + f), 1e-9, 1.5); return math.sin(p)
    except Exception:
        return None
def fmin(k, m, c, two=True, rmin2=5):
    best = None
    for r1 in range(4, k):
        for r2 in ([None] + list(range(rmin2, k - r1))) if two else [None]:
            def g(f):
                s1 = s(r1, f)
                if s1 is None: return -1
                W = (k + f) * s1 + 2 * c * (r1 + f)
                if r2 is None:
                    W += f * (k - r1)
                else:
                    s2 = s(r2, f)
                    if s2 is None: return -1
                    W += (k - r1) * s2 + 2 * c * (r2 + f)
                return 2 * k * f + f * f - m - W
            fs = [0.3 + 0.7 * i / 140 for i in range(141)]
            ok = [f for f in fs if g(f) >= 0]
            if not ok: continue
            lo = max(0.3, ok[0] - 0.005)
            f = brentq(g, lo, ok[0]) if g(lo) < 0 else ok[0]
            if best is None or f < best[0]: best = (f, r1, r2)
    return best
if __name__ == '__main__':
    R = {int(x['n']): x for x in csv.DictReader(open('/mnt/project-files/squares/leaderboard-20261103/gap_vs_best_per_n.csv'))}
    for c in [float(a) for a in sys.argv[1].split(',')]:
        cnt = 0; tot = 0; lead = 0
        for n in range(200, 1001):
            k = math.isqrt(n); m = n - k * k
            if m == 0 or m > 0.6 * (2 * k + 1): continue
            b = fmin(k, m, c)
            if b is None: continue
            ours = float(R[n]['our_L']) - k; bl = float(R[n]['best_L']) - k
            if b[0] < ours - 1e-6:
                cnt += 1; tot += (ours - b[0]) / math.sqrt(n)
            if b[0] < bl: lead += 1
        print(f'c={c}: beats ours at {cnt} n, score gain {tot:.4f}; beats leaders at {lead} n', flush=True)
