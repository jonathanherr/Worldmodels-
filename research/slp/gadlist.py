import sys, json, numpy as np
sys.path.insert(0, '/home/claude/worldmodels-')
from wm import fault_dp as fd
def load(src, n):
    rows = json.load(open(src))[n]
    return np.array([[float(r['x']), float(r['y']), float(r['angle'])] for r in rows])
if __name__ == '__main__':
    p = load(sys.argv[1], sys.argv[2]); fmax = float(sys.argv[3])
    best = {}
    for g in fd.extract(0, p, max_wh=31):
        if g.f < fmax and g.d >= 2:
            k = (g.w, g.h, g.d)
            if k not in best or g.f < best[k].f: best[k] = g
    # keep only Pareto (w,h,d): no other with w'<=w,h'<=h,d'>=d and f'<=f
    items = sorted(best.values(), key=lambda g: g.f)
    print(len(items))
    for g in items[:40]: print(g.w, g.h, g.d, round(g.f, 5))
