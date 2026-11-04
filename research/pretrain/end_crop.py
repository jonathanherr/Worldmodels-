"""Band-end cropper: find full-width tilted bands in a packing and measure their end waste.

A band is a cluster of squares sharing one tilt (mod 90 deg) whose centres span most of the box in
one direction. For a horizontal band occupying the slab y0 <= y <= y1 (H = y1 - y0):

    slab waste      = H * L - (area of all squares clipped to the slab)
    interior rate   = waste per unit length measured in the middle 50% of the band
    E (both ends)   = slab waste - L * interior rate
    c (per row/end) = E / (2 H)                      (same definition as research/corner_scale)

Vertical bands are handled by swapping x and y. Each end is also cropped into band coordinates
(wall at u=0, band bottom at v=0, right ends mirrored) for the pretraining corpus.

Only bands with span > 0.8 and purity > 0.85 go into ends.csv (wall-to-wall bands).

Usage: python research/pretrain/end_crop.py --csv SUBMISSION.csv --certs DIR --out OUT_DIR
"""
import argparse, csv, json, math, os, sys, collections
from fractions import Fraction
import numpy as np

TILT_MIN = 2.0      # degrees; below this a square counts as axis-aligned (polish wobble)
MIN_BAND = 12       # squares in a tilt cluster before it can be a band
SPAN = 0.6          # band centres must span this fraction of L along the band


def corners(x, y, a):
    c, s = np.cos(a), np.sin(a)
    d = np.array([[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]])
    px = x[:, None] + d[None, :, 0] * c[:, None] - d[None, :, 1] * s[:, None]
    py = y[:, None] + d[None, :, 0] * s[:, None] + d[None, :, 1] * c[:, None]
    return px, py


def clip_area(poly, x0, x1, y0, y1):
    """Area of convex polygon (list of (x,y)) clipped to the rectangle."""
    def clip(pts, axis, val, keep_ge):
        out = []
        m = len(pts)
        for i in range(m):
            p, q = pts[i], pts[(i + 1) % m]
            pin = (p[axis] >= val) if keep_ge else (p[axis] <= val)
            qin = (q[axis] >= val) if keep_ge else (q[axis] <= val)
            if pin:
                out.append(p)
            if pin != qin:
                t = (val - p[axis]) / (q[axis] - p[axis])
                out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
        return out
    pts = poly
    for axis, val, ge in ((0, x0, True), (0, x1, False), (1, y0, True), (1, y1, False)):
        pts = clip(pts, axis, val, ge)
        if not pts:
            return 0.0
    a = 0.0
    for i in range(len(pts)):
        p, q = pts[i], pts[(i + 1) % len(pts)]
        a += p[0] * q[1] - q[0] * p[1]
    return abs(a) / 2


def region_fill(px, py, x0, x1, y0, y1):
    sel = np.where((px.max(1) > x0) & (px.min(1) < x1) & (py.max(1) > y0) & (py.min(1) < y1))[0]
    return sum(clip_area(list(zip(px[i], py[i])), x0, x1, y0, y1) for i in sel)


def find_bands(n, L, x, y, a):
    t = np.mod(np.degrees(a), 90.0)
    tilt = np.minimum(t, 90 - t)
    # cluster tilts (mod 90): sort, split at gaps > 1 degree
    key = np.full(len(t), -1.0)
    ti = np.where(tilt > TILT_MIN)[0]
    if len(ti):
        order = ti[np.argsort(t[ti])]
        grp = [[order[0]]]
        for i in order[1:]:
            (grp[-1].append(i) if t[i] - t[grp[-1][-1]] < 1.0 else grp.append([i]))
        for g in grp:
            key[g] = float(np.median(t[g]))
    out = []
    px, py = corners(x, y, a)
    for k, cnt in collections.Counter(key[key >= 0]).items():
        if cnt < MIN_BAND:
            continue
        idx = np.where(key == k)[0]
        for orient in ('h', 'v'):
            U, V = (x, y) if orient == 'h' else (y, x)
            PU, PV = (px, py) if orient == 'h' else (py, px)
            if U[idx].max() - U[idx].min() < SPAN * L:
                continue
            v0, v1 = PV[idx].min(), PV[idx].max()
            H = v1 - v0
            if H > 0.6 * L or H < 1.5:
                continue  # a block or a diagonal strip, not a band
            # waste in the slab (work in (u, v) coordinates)
            slab = region_fill(PU, PV, 0.0, L, v0, v1)
            u0, u1 = U[idx].min(), U[idx].max()
            w0, w1 = u0 + 0.25 * (u1 - u0), u1 - 0.25 * (u1 - u0)
            mid = region_fill(PU, PV, w0, w1, v0, v1)
            inmid = np.where((U > w0) & (U < w1) & (V > v0) & (V < v1))[0]
            purity = float(np.isin(inmid, idx).mean()) if len(inmid) else 0.0
            # tilted squares of this cluster lying above/below the slab middle = not a single band
            rate = ((w1 - w0) * H - mid) / (w1 - w0)
            sphi = math.sin(math.radians(min(k, 90 - k)))
            waste = L * H - slab
            E = waste - L * rate
            c = E / (2 * H)
            c_eff = (waste - L * sphi) / (2 * H)  # all slab waste beyond an ideal column interior, per row per end
            ends = []
            half = {}
            for side, (h0, h1) in (('lo', (0.0, L / 2)), ('hi', (L / 2, L))):
                hw = (h1 - h0) * H - region_fill(PU, PV, h0, h1, v0, v1)
                half[side] = (hw - (h1 - h0) * sphi) / H  # this end's waste per row, ideal interior
            for side in ('lo', 'hi'):
                win = H + 2.0
                if side == 'lo':
                    sel = np.where((U < win) & (V > v0 - 0.5) & (V < v1 + 0.5))[0]
                    uu = U[sel]
                    ang = a[sel]
                else:
                    sel = np.where((U > L - win) & (V > v0 - 0.5) & (V < v1 + 0.5))[0]
                    uu = L - U[sel]
                    ang = -a[sel]  # mirror
                vv = V[sel] - v0
                if orient == 'v':
                    ang = (math.pi / 2 - ang) if side == 'lo' else (ang - math.pi / 2)
                t_end = np.minimum(np.mod(np.degrees(ang), 90), 90 - np.mod(np.degrees(ang), 90))
                ends.append({'side': side, 'c_end': float(half[side]), 'wall_end': bool(np.all((t_end < TILT_MIN) | (abs(t_end - min(k, 90 - k)) < 1.0))), 'u': uu.round(9).tolist(), 'v': vv.round(9).tolist(),
                             'angle': np.mod(ang, math.pi / 2).round(9).tolist()})
            out.append({'n': int(n), 'L': float(L), 'k': int(math.isqrt(n)), 'm': int(n - math.isqrt(n) ** 2), 'orient': orient,
                        'tilt_deg': float(min(k, 90 - k)), 'band_squares': int(cnt), 'H': float(H),
                        'v0': float(v0), 'interior_rate': float(rate), 'slab_waste': float(waste),
                        'E': float(E), 'c': float(c), 'c_eff': float(c_eff), 'purity': purity, 'span': float((u1 - u0) / L),
                        'clean': bool(purity > 0.97 and (u1 - u0) > 0.8 * L and 0 < rate < math.sin(math.radians(min(k, 90 - k))) + 0.15), 'other_tilts': int((tilt > TILT_MIN).sum() - cnt),
                        'total_waste': float(L * L - n), 'ends': ends})
    return out


def load_csv(path):
    rows = collections.defaultdict(list)
    with open(path) as f:
        r = csv.reader(f); next(r)
        for _, n, i, x, y, a in r:
            rows[int(n)].append((float(x), float(y), float(a)))
    for n, s in rows.items():
        A = np.array(s)
        # recover L from the squares' extent
        px, py = corners(A[:, 0], A[:, 1], A[:, 2])
        L = max(px.max(), py.max())
        yield n, L, A[:, 0], A[:, 1], A[:, 2]


def load_cert(path):
    L = [l.split() for l in open(path) if l.strip() and not l.startswith('#')]
    n = int(L[0][0]); side = float(Fraction(L[0][1]))
    v = np.array([[float(Fraction(a)), float(Fraction(b)), 2 * math.atan(float(Fraction(t)))] for a, b, t in L[1:n + 1]])
    return n, side, v[:, 0], v[:, 1], v[:, 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv'); ap.add_argument('--certs'); ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    jobs = []
    if args.csv:
        jobs += [('ours', p) for p in load_csv(args.csv)]
    res = []
    for src, (n, L, x, y, a) in jobs:
        for b in find_bands(n, L, x, y, a):
            b['source'] = src; res.append(b)
    if args.certs:
        for fn in sorted(os.listdir(args.certs)):
            try:
                n, L, x, y, a = load_cert(os.path.join(args.certs, fn))
            except Exception as e:
                print('skip', fn, e, file=sys.stderr); continue
            for b in find_bands(n, L, x, y, a):
                b['source'] = 'couzo:' + fn; res.append(b)
            print(fn, n, len(res), file=sys.stderr, flush=True)
    with open(os.path.join(args.out, 'ends.jsonl'), 'w') as f:
        for b in res:
            f.write(json.dumps(b) + '\n')
    with open(os.path.join(args.out, 'bands.csv'), 'w') as f:
        w = csv.writer(f)
        w.writerow(['source', 'n', 'k', 'm', 'L', 'orient', 'tilt_deg', 'band_squares', 'H', 'c', 'c_eff', 'E', 'interior_rate', 'purity', 'span', 'clean', 'other_tilts'])
        for b in sorted(res, key=lambda b: b['c']):
            w.writerow([b['source'], b['n'], b['k'], b['m'], round(b['L'], 6), b['orient'], b['tilt_deg'], b['band_squares'],
                        round(b['H'], 4), round(b['c'], 4), round(b['c_eff'], 4), round(b['E'], 4), round(b['interior_rate'], 4), round(b['purity'], 3), round(b['span'], 3), int(b['clean']), b['other_tilts']])
    with open(os.path.join(args.out, 'ends.csv'), 'w') as f:
        w = csv.writer(f)
        w.writerow(['source', 'n', 'k', 'm', 'f', 'orient', 'side', 'tilt_deg', 'H', 'c_end', 'wall_end', 'squares'])
        rows = [(b, e) for b in res if b['span'] > 0.8 and b['purity'] > 0.85 for e in b['ends']]
        for b, e in sorted(rows, key=lambda t: t[1]['c_end']):
            w.writerow([b['source'], b['n'], b['k'], b['m'], round(b['L'] - b['k'], 6), b['orient'], e['side'],
                        round(b['tilt_deg'], 3), round(b['H'], 4), round(e['c_end'], 4), int(e['wall_end']), len(e['u'])])
    print('bands', len(res))


if __name__ == '__main__':
    main()
