import sys, math, time, json
import numpy as np
sys.path.insert(0, '/home/claude/worldmodels-')
from research.corner_scale.cut_band import load_cert, cut
from wm import pack_opt
cert, k, c, x0 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
slabs = [tuple(map(float, s.split(':'))) for s in sys.argv[5].split(',')]
out = sys.argv[6]
n0, L0, P0 = load_cert(cert)
D = int(L0) - k
P, L = cut(P0, L0, D, c, x0, slabs)
target = k * k + 1
x, y, t = P[:, 0].copy(), P[:, 1].copy(), P[:, 2].copy()
# drop extras: squares with largest overlap/wall penetration energy
while len(x) > target:
    e, gx, gy, gt, gL, worst = pack_opt.energy_grad(x, y, t, L, 0.0)
    gnorm = gx**2 + gy**2 + gt**2
    i = int(np.argmax(gnorm))
    x, y, t = np.delete(x, i), np.delete(y, i), np.delete(t, i)
print('count', len(x), 'target', target, 'start L', L, flush=True)
if len(x) < target or '--count' in sys.argv:
    sys.exit('stop')
t0 = time.time()
r = pack_opt.compact_squeeze(x, y, t, L)
if r is None:
    sys.exit('fail')
x, y, t, L2 = r
print(f'k={k} c={c} L={L2:.6f} f={L2-k:.6f} feasible={pack_opt.feasible(x,y,t,L2)} {time.time()-t0:.0f}s', flush=True)
json.dump({'n': len(x), 'L': L2, 'x': x.tolist(), 'y': y.tolist(), 't': t.tolist()}, open(out, 'w'))
