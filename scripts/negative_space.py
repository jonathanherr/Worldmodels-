"""Compute the negative space (empty region) of each packing in a submission CSV.

Squares are unioned (slightly grown to close the ~1e-9 export gaps), subtracted
from the bounding square, and the leftover holes are written as simplified
polygons plus per-hole stats (area, perimeter, touches wall, thinness).
"""
import argparse, json, math, sys
from collections import defaultdict

from shapely.geometry import Polygon, box
from shapely.ops import unary_union

GROW = 2e-6      # closes export gaps between touching squares
MIN_AREA = 1e-5  # ignore crumbs


def corners(x, y, a):
    c, s, h = math.cos(a), math.sin(a), 0.5
    return [(x + h * c - h * s, y + h * s + h * c), (x - h * c - h * s, y - h * s + h * c),
            (x - h * c + h * s, y - h * s - h * c), (x + h * c + h * s, y + h * s - h * c)]


def holes_for(sq):
    pts = [corners(*q) for q in sq]
    xs = [p[0] for c in pts for p in c]; ys = [p[1] for c in pts for p in c]
    x0, y0 = min(xs), min(ys)
    L = max(max(xs) - x0, max(ys) - y0)
    polys = [Polygon([(px - x0, py - y0) for px, py in c]).buffer(GROW, join_style=2) for c in pts]
    free = box(0, 0, L, L).difference(unary_union(polys))
    geoms = getattr(free, "geoms", [free])
    out = []
    for g in geoms:
        if g.is_empty or g.area < MIN_AREA:
            continue
        g = g.simplify(1e-4)
        bx = g.bounds
        wall = bx[0] < 1e-3 or bx[1] < 1e-3 or bx[2] > L - 1e-3 or bx[3] > L - 1e-3
        out.append({"a": g.area, "p": g.length, "wall": wall,
                    "xy": [[round(px, 4), round(py, 4)] for px, py in g.exterior.coords[:-1]]})
    out.sort(key=lambda h: -h["a"])
    return L, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv"); ap.add_argument("out"); ap.add_argument("--nmax", type=int, default=1000)
    a = ap.parse_args()
    by = defaultdict(list)
    with open(a.csv) as f:
        next(f)
        for line in f:
            _, n, _, x, y, ang = line.strip().split(",")
            n = int(n)
            if n <= a.nmax:
                by[n].append((float(x), float(y), float(ang)))
    res = {}
    for n in sorted(by):
        sq = by[n]
        L, hs = holes_for(sq)
        tilted = sum(1 for q in sq if abs(math.remainder(q[2], math.pi / 2)) > 1e-6)
        res[n] = {"L": L, "tilted": tilted, "holes": hs}
        if n % 50 == 0:
            print(n, round(L, 4), len(hs), file=sys.stderr)
    json.dump(res, open(a.out, "w"), separators=(",", ":"))


if __name__ == "__main__":
    main()
