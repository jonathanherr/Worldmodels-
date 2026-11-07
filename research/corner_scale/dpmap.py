import sys, math, json
import numpy as np
sys.path.insert(0, '/home/claude/worldmodels-')
from research.corner_scale.dpbuild import best_dp
out = {}
for f in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
    for r in range(5, 21):
        try:
            w, S, J, N, kind = best_dp(r, f, na=21, noff=5)
        except Exception as e:
            continue
        out[f'{r},{f}'] = w
        print(r, f, round(w, 3), round(w / S.H, 4), kind, flush=True)
        json.dump(out, open(sys.argv[1], 'w'))
