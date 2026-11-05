"""Grow a tilted-block gadget by whole lattice periods.

The tilted part of a gadget is a patch of a square lattice at angle theta; its horizontal period is
P = 1/sin(theta') for the lattice vector closest to horizontal. Cut the gadget at x = c, shift
everything right of the cut by the period (tilted squares) or by the nearest integer (axis squares),
copy the strip left of the cut into the gap, and let SLP absorb the small mismatch. Same for y.
"""
import math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from slp import slp
from gadget import hop


def lattice_period(theta, axis, maxk=12):
    """Shortest lattice vectors near the given axis: list of (length, mismatch to integer)."""
    u = np.array([math.cos(theta), math.sin(theta)]); v = np.array([-math.sin(theta), math.cos(theta)])
    out = []
    for a in range(-maxk, maxk + 1):
        for b in range(-maxk, maxk + 1):
            p = a * u + b * v
            along, across = (p[0], p[1]) if axis == 0 else (p[1], p[0])
            if along > 0.5 and abs(across) < 1e-9 + 0.02:
                out.append((along, across, a, b))
    out.sort()
    return out


def grow(q, w, h, axis, c, P, across, m, tilt_mask):
    """Insert one period at coordinate c along axis (0 = x, 1 = y)."""
    q = q.copy()
    k = axis
    right = q[:, k] > c
    shift = np.where(tilt_mask, P, m)
    new = []
    # copy the strip [c - P, c] (tilted) / [c - m, c] (axis) into the gap
    for i in range(len(q)):
        lo = c - (P if tilt_mask[i] else m)
        if lo < q[i, k] <= c:
            r = q[i].copy(); r[k] += (P if tilt_mask[i] else m)
            if tilt_mask[i]:
                r[1 - k] += across
            new.append(r)
    q[right, k] += shift[right]
    q[right & tilt_mask, 1 - k] += across
    out = np.vstack([q, np.array(new).reshape(-1, 3)])
    return out


if __name__ == '__main__':
    src, w, h = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    q = np.load(src)
    ang = np.mod(q[:, 2], math.pi / 2)
    tilt = (ang > 0.05) & (ang < math.pi / 2 - 0.05)
    theta = float(np.median(ang[tilt]))
    print('theta', math.degrees(theta), 'tilted', tilt.sum(), 'of', len(q))
    for axis in (0, 1):
        per = [p for p in lattice_period(theta, axis) if p[0] < 10][:6]
        for P, across, a, b in per:
            m = int(round(P))
            span = q[:, axis]
            for c in np.linspace(span.min() + 2, span.max() - 2, 5):
                g = grow(q, w, h, axis, c, P, across, m, tilt)
                w2, h2 = (w + m, h) if axis == 0 else (w, h + m)
                # drop overlaps greedily (tilted first), then refill holes with axis squares
                from gen_block import overlaps, inside
                W2, H2 = w2 + 1.0, h2 + 1.0
                keep = []
                for r in g[np.argsort(~np.r_[tilt, np.ones(len(g) - len(tilt), bool)] if False else np.zeros(len(g)))]:
                    if inside(tuple(r), W2, H2) and not overlaps(np.array(keep).reshape(-1, 3), tuple(r), tol=-1e-6):
                        keep.append(r)
                for gx in range(w2 + 1):
                    for gy in range(h2 + 1):
                        for r in ((gx + .5, gy + .5, 0.), (W2 - gx - .5, H2 - gy - .5, 0.)):
                            if inside(r, W2, H2) and not overlaps(np.array(keep).reshape(-1, 3), r):
                                keep.append(np.array(r))
                g = np.array(keep)
                d2 = len(g) - w2 * h2
                x, y, t, L, v, L0 = slp(g[:, 0], g[:, 1], g[:, 2], iters=300, ex=w2, ey=h2)
                tag = 'OK' if v < 1e-10 else 'viol %.1e' % v
                print(f'axis {axis} P {P:.3f} m {m} c {c:.1f}: {w2}x{h2}+{d2} f {L:.6f} {tag}', flush=True)
                if v < 1e-10:
                    np.save(f'{os.path.dirname(src) or "."}/st_{w2}_{h2}_{d2}_{axis}_{P:.2f}_{c:.1f}.npy', np.c_[x, y, t])
