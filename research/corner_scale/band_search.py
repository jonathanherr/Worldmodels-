import math, sys, itertools, json
import numpy as np
sys.path.insert(0, '/home/claude/worldmodels-')
from research.corner_scale.band import Band, square_poly, fill_rows

J = 5  # end columns per side with free (lo, hi, align)

def col_squares(B, X, lo, hi, align):
    slide = 0.0 if align == 0 else (-lo if align == 1 else B.m - hi)
    return B.column(X, lo, hi, slide)

def end_count(B, X0, params, offset, side):
    """Left-frame end: columns j=0..J-1 at X0 + j p with params, columns J..J+2 full (blockers).
    Returns (count of squares in end columns + rows, squares list). side only labels."""
    sq = []
    for j in range(J):
        lo, hi, al = params[j]
        for (x, y, a) in col_squares(B, X0 + j * B.p, lo, hi, al):
            poly = square_poly(x, y, a)
            if poly[:, 0].min() < -1e-12:
                continue
            sq.append((x, y, a))
    blockers = [square_poly(*q) for q in sq]
    for j in range(J, J + 3):
        blockers += [square_poly(*q) for q in col_squares(B, X0 + j * B.p, 0, B.m, 0)]
    rows = fill_rows(blockers, B.H, 1e9, 'L', offset, B.r + 1)
    return len(sq) + len(rows), sq + rows

def opt_end(B, X0, iters=3):
    m = B.m
    params = [[0, m, 0] for _ in range(J)]
    offs = [0.0, B.f / 2, B.f]
    best = (-1, None, None)
    for off in offs:
        cur = [p[:] for p in params]
        val, _ = end_count(B, X0, cur, off, 'L')
        for _ in range(iters):
            improved = False
            for j in range(J):
                for lo in range(0, m + 1):
                    for hi in range(lo, m + 1):
                        for al in (0, 1, 2):
                            if (hi - lo == 0 or hi - lo == m) and al != 0:
                                continue
                            trial = [p[:] for p in cur]
                            trial[j] = [lo, hi, al]
                            v, _ = end_count(B, X0, trial, off, 'L')
                            if v > val:
                                val, cur, improved = v, trial, True
            if not improved:
                break
        if val > best[0]:
            best = (val, cur, off)
    return best

if __name__ == '__main__':
    r, f = int(sys.argv[1]), float(sys.argv[2])
    B = Band(r, f, 100.0)
    print('phi deg', math.degrees(B.phi), 'pitch', B.p)
    # gadget value relative to full columns: g = count - m * J, as function of X0 in [0, p)
    res = []
    for X0 in np.linspace(0, B.p, 13)[:-1] + 0.5:
        val, params, off = opt_end(B, X0, iters=2)
        res.append((X0, val, params, off))
        print(f'X0={X0:.3f} end count={val}  (J*m={J*B.m})  params={params} off={off:.3f}', flush=True)
