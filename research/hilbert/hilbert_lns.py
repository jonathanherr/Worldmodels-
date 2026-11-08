"""Hilbert-curve large-neighbourhood search (destroy & repair) on finished square packings.

Squares are ordered by the Hilbert index of their centres. A contiguous run of m squares along
the curve is a compact 2D patch (runs of 4^k aligned with the curve are exact sub-quadrants), so
the curve gives multiscale, locality-preserving neighbourhoods for free. Each trial:
  1. pick a window of m squares (Hilbert run, or a control: kNN ball / random scatter),
  2. destroy it: re-scatter those squares at random in the window's bounding box, random angles,
  3. shrink the box slightly (target L*(1-eps)), repair with pack_opt.compact (L free) + squeeze,
  4. keep the result if it is feasible and smaller than the incumbent.
"""
import argparse, csv, json, os, sys, time
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from wm import pack_opt  # noqa: E402


def hilbert_d(order, x, y):
    """Hilbert index of integer cell (x, y) on a 2^order grid (classic xy2d)."""
    n = 1 << order
    d = 0
    s = n >> 1
    while s > 0:
        rx = 1 if (x & s) else 0
        ry = 1 if (y & s) else 0
        d += s * s * ((3 * rx) ^ ry)
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x
        s >>= 1
    return d


def hilbert_order(x, y, L, order=10):
    g = (1 << order) - 1
    ix = np.clip((x / L * g).astype(int), 0, g)
    iy = np.clip((y / L * g).astype(int), 0, g)
    return np.argsort([hilbert_d(order, int(a), int(b)) for a, b in zip(ix, iy)], kind="stable")


def load(csv_path, ns):
    out = {n: [] for n in ns}
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            n = int(r["n"])
            if n in out:
                out[n].append((float(r["x"]), float(r["y"]), float(r["angle"])))
    res = {}
    for n, rows in out.items():
        a = np.array(rows)
        x, y, t = a[:, 0], a[:, 1], a[:, 2]
        w = 0.5 * (np.abs(np.cos(t)) + np.abs(np.sin(t)))
        x0, y0 = (x - w).min(), (y - w).min()
        L = max((x + w).max() - x0, (y + w).max() - y0)
        res[n] = (x - x0, y - y0, t, L)
    return res


def pick(method, x, y, L, m, rng):
    n = len(x)
    if method == "hilbert":
        order = hilbert_order(x, y, L)
        s = rng.integers(0, n - m + 1)
        return order[s:s + m]
    if method == "knn":
        c = rng.integers(n)
        d = (x - x[c]) ** 2 + (y - y[c]) ** 2
        return np.argsort(d)[:m]
    return rng.choice(n, m, replace=False)


def repair_window(x, y, t, L, W, maxiter=1500):
    """Minimise overlap energy over the window squares only (the rest frozen) at fixed side L."""
    from scipy.optimize import minimize
    n = len(x)
    mask = np.zeros(n, bool); mask[W] = True
    def f(z):
        xx, yy, tt = x.copy(), y.copy(), t.copy()
        xx[W], yy[W], tt[W] = z[:len(W)], z[len(W):2 * len(W)], z[2 * len(W):]
        e, gx, gy, gt, _, _ = pack_opt.energy_grad(xx, yy, tt, L, pack_opt.MARGIN)
        return e, np.concatenate([gx[W], gy[W], gt[W]])
    z = minimize(f, np.concatenate([x[W], y[W], t[W]]), jac=True, method="L-BFGS-B",
                 options={"maxiter": maxiter, "ftol": 0.0, "gtol": 1e-14}).x
    x, y, t = x.copy(), y.copy(), t.copy()
    x[W], y[W], t[W] = z[:len(W)], z[len(W):2 * len(W)], z[2 * len(W):]
    return x, y, t


def finish(x, y, t, L):
    """Smallest feasible side near L: inflate in fine steps until relaxed-feasible, then squeeze."""
    for d in (0.0, 1e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2):
        Lt = L * (1 + d)
        k = Lt / L
        rx, ry, rt, _ = pack_opt.relax_lbfgs(x * k, y * k, t, Lt, maxiter=3000)
        if pack_opt.feasible(rx, ry, rt, Lt):
            return pack_opt.squeeze(rx, ry, rt, Lt, rounds=25, step=max(d, 1e-6) / 2, method="lbfgs", maxiter=1500)
    return None


def trial(args):
    n, method, m, seed, (x0, y0, t0_, L), eps, R = args
    rng = np.random.default_rng(seed)
    W = pick(method, x0, y0, L, m, rng)
    lo_x, hi_x = x0[W].min() - 0.3, x0[W].max() + 0.3
    lo_y, hi_y = y0[W].min() - 0.3, y0[W].max() + 0.3
    Lt = L * (1 - eps)
    k = Lt / L
    tic = time.time()
    best = None
    for r in range(R):
        x, y, t = x0 * k, y0 * k, t0_.copy()
        x[W] = rng.uniform(lo_x, hi_x, m) * k
        y[W] = rng.uniform(lo_y, hi_y, m) * k
        t[W] = rng.choice(np.concatenate([t0_, [0.0, np.pi / 4]]), m) + rng.normal(0, 0.1, m)
        x, y, t = repair_window(x, y, t, Lt, W)
        e = pack_opt.energy_grad(x, y, t, Lt, 0.0)[0]
        moved = float(np.sqrt(((x[W] - x0[W] * k) ** 2 + (y[W] - y0[W] * k) ** 2).mean()))
        if best is None or e < best[0]:
            best = (e, x, y, t, moved)
    e, x, y, t, moved = best
    out = dict(n=n, method=method, m=m, seed=seed, L0=float(L), e=float(e), moved=moved, L=None)
    if e < 1e-8:
        r = finish(x, y, t, Lt)
        if r is not None:
            sx, sy, st, sL = r
            out["L"] = float(sL)
            if sL < L - 1e-9:
                out["sol"] = [sx.tolist(), sy.tolist(), st.tolist()]
    out["sec"] = time.time() - tic
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--ns", required=True)
    ap.add_argument("--methods", default="hilbert,knn,random")
    ap.add_argument("--ms", default="8,16")
    ap.add_argument("--trials", type=int, default=6)
    ap.add_argument("--eps", type=float, default=2e-3)
    ap.add_argument("--restarts", type=int, default=20)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ns = [int(v) for v in a.ns.split(",")]
    packs = load(a.csv, ns)
    jobs = [(n, meth, int(m), 1000 * i + 7 * n + int(m), packs[n], a.eps, a.restarts)
            for n in ns for meth in a.methods.split(",") for m in a.ms.split(",") for i in range(a.trials)]
    with Pool(a.procs) as p, open(a.out, "w") as f:
        for r in p.imap_unordered(trial, jobs):
            f.write(json.dumps(r) + "\n"); f.flush()
            print(r["n"], r["method"], r["m"], "e=%.1e mv=%.2f" % (r["e"], r["moved"]), r["L0"], r["L"], round(r["sec"], 1), "WIN" if r.get("sol") else "", flush=True)
