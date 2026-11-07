import sys, math, time, json
import numpy as np
sys.path.insert(0, '/home/claude/worldmodels-')
from research.corner_scale.cut_band import load_cert, cut, band_mask
from wm import pack_opt
n0, L0, P0 = load_cert(sys.argv[1])
k0 = int(L0)
for k in [int(a) for a in sys.argv[2].split(',')]:
    D = k0 - k
    target = k * k + 1
    best = None
    for c in [D, D - 1, D + 1]:
        for x0 in [20.0]:
            for y0 in [0.0, 30.0 - 0.0]:
                P, L = cut(P0, L0, D, c, x0, y0 if y0 < 1 else L0 - D - 0.0)
                cnt = len(P)
                print(f"k={k} D={D} c={c} y0={y0} count={cnt} target={target}", flush=True)
