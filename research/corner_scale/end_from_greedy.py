import sys, math, json
import numpy as np
sys.path.insert(0, '/home/claude/worldmodels-')
from research.corner_scale.band import Band, square_poly
from research.corner_scale.band_search import end_count, opt_end, J, col_squares
from research.corner_scale.endgadget import Gadget

def greedy_gadget(r, f, X0):
    B = Band(r, f, 100.0)
    val, params, off = opt_end(B, X0, iters=2)
    _, sq = end_count(B, X0, params, off, 'L')
    sq = np.array(sq)
    # face: left face of column J -> c = min over its squares of u . vertex
    G = Gadget(r, f, B.phi)
    colJ = col_squares(B, X0 + J * B.p, 0, B.m, 0)
    c = min((square_poly(*q) @ np.array([G.cs, -G.sn])).min() for q in colJ)
    return G, sq, c

if __name__ == '__main__':
    r, f, X0 = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
    G, sq, c = greedy_gadget(r, f, X0)
    x, y, t = sq[:, 0], sq[:, 1], sq[:, 2]
    print('N', len(sq), 'c', c, 'feasible', G.feasible(x, y, t, c + 1e-9), 'waste', G.waste(len(sq), c), 'per row', G.waste(len(sq), c) / G.H)
    cx, cy, ct, cc = G.compact(x, y, t, c)
    res = G.squeeze(cx, cy, ct, cc + 0.002)
    if res:
        x2, y2, t2, c2 = res
        print('after compact: c', c2, 'waste', G.waste(len(sq), c2), 'per row', G.waste(len(sq), c2) / G.H)
