"""Import public certified record packings (Francisco Couzo's square-packing repo) into the store.

    git clone --depth 1 https://github.com/franciscouzo/square-packing /tmp/couzo
    SQUARES_BEST_DIR=... python -m scripts.import_certs /tmp/couzo/certificates/best --max-n 1000

Each .cert file is a header line "n side" then n lines "x y tan(theta/2)" of exact rationals
(square centres). The packing is dilated about the box centre by 1 + 1e-12 so exactly touching
squares get a gap the official interval-arithmetic metric can certify, then offered to the
store, which keeps it only if the metric accepts it and it beats the stored side.
"""

import argparse
import math
from fractions import Fraction
from pathlib import Path

import numpy as np

from wm import best_store


def load_cert(path: Path, dilation: float = 1e-12):
    """(n, side, x, y, deg) from a certificate file, dilated about the box centre."""
    rows = [line.split() for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")]
    n, side = int(rows[0][0]), float(Fraction(rows[0][1]))
    x = np.array([float(Fraction(r[0])) for r in rows[1:n + 1]])
    y = np.array([float(Fraction(r[1])) for r in rows[1:n + 1]])
    deg = np.array([math.degrees(2 * math.atan(float(Fraction(r[2])))) for r in rows[1:n + 1]])
    c = side / 2
    return n, side, c + (x - c) * (1 + dilation), c + (y - c) * (1 + dilation), deg


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cert_dir", type=Path)
    p.add_argument("--max-n", type=int, default=best_store.MAX_N)
    args = p.parse_args()
    files = sorted(args.cert_dir.glob("n*.cert"), key=lambda f: int(f.stem[1:]))
    won = 0
    for f in files:
        if int(f.stem[1:]) > args.max_n:
            continue
        n, side, x, y, deg = load_cert(f)
        r = best_store.offer(n, x, y, deg, f"public-cert:{f.name}")
        won += r["accepted"]
        print(f"n={n} cert {side:.10f} -> {r['reason']} ({r['side']})", flush=True)
    print(f"accepted {won}; store total {best_store.total_score():.6f}")
