"""Batched perturb-and-relax polish for square packings (the loop the public record-setters use).

Each round takes the current best packing of n squares at side L0 and makes B variants:
local melts (jitter the squares in a disc), rotated patches (turn a disc of squares about its
centre) and small global kicks. Every variant is shrunk into a box slightly below L0 and relaxed
in one batched PyTorch run (float32 on Apple MPS or CUDA, float64 on CPU) on the separating-axis
overlap energy of `wm.pack_opt`. The variants that end with the least overlap go to the float64
numba `squeeze`, and anything smaller than the record is offered to the best-per-n store, which
certifies it with the official metric.

The torch energy uses a fixed pair list taken from the seed (centres closer than `pair_radius`),
so it is only a screen: perturbations move squares by well under a unit and the float64 polish
re-checks every pair.
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch

from . import best_store
from .pack_opt import MARGIN, feasible, relax_lbfgs, squeeze


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_pairs(x: np.ndarray, y: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
    """Index pairs (i < j) whose seed centres are closer than radius, via a cell grid."""
    cells: dict[tuple[int, int], list[int]] = {}
    for k, (a, b) in enumerate(zip(x, y)):
        cells.setdefault((int(a // radius), int(b // radius)), []).append(k)
    I, J = [], []
    r2 = radius * radius
    for (cx, cy), members in cells.items():
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in cells.get((cx + dx, cy + dy), ()):
                    for i in members:
                        if i < j and (x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2 < r2:
                            I.append(i)
                            J.append(j)
    return np.array(I, dtype=np.int64), np.array(J, dtype=np.int64)


def batch_energy(x, y, t, L, I, J, margin: float = MARGIN):
    """Overlap energy of B packings at once. x, y, t: (B, n); L: (B,). Returns (B,) energy, (B,) worst depth."""
    dx = x[:, J] - x[:, I]
    dy = y[:, J] - y[:, I]
    ci, si = torch.cos(t[:, I]), torch.sin(t[:, I])
    cj, sj = torch.cos(t[:, J]), torch.sin(t[:, J])
    delta = t[:, J] - t[:, I]
    rad = 0.5 + 0.5 * (delta.cos().abs() + delta.sin().abs()) + margin
    proj = torch.stack([dx * ci + dy * si, -dx * si + dy * ci, dx * cj + dy * sj, -dx * sj + dy * cj])
    depth = (rad - proj.abs()).min(dim=0).values  # (B, P): SAT penetration depth
    e = depth.clamp(min=0).pow(2).sum(dim=1)
    w = 0.5 * (t.cos().abs() + t.sin().abs())
    Lb = L[:, None]
    walls = ((w - x).clamp(min=0).pow(2) + (x + w - Lb).clamp(min=0).pow(2)
             + (w - y).clamp(min=0).pow(2) + (y + w - Lb).clamp(min=0).pow(2)).sum(dim=1)
    worst = torch.maximum(depth.max(dim=1).values,
                          torch.stack([(w - x), (x + w - Lb), (w - y), (y + w - Lb)]).amax(dim=(0, 2)))
    return e + walls, worst


def perturb(x, y, t, L, gen: torch.Generator, kinds=("melt", "rotate", "kick")):
    """In place: give each of the B rows one random local move. x, y, t: (B, n); L: (B,)."""
    B, n = x.shape
    dev = x.device
    u = lambda *s: torch.rand(*s, generator=gen, device="cpu").to(dev, x.dtype)
    g = lambda *s: torch.randn(*s, generator=gen, device="cpu").to(dev, x.dtype)
    kind = torch.randint(len(kinds), (B,), generator=gen).to(dev)
    cx, cy = u(B) * L, u(B) * L
    r = 1.0 + 3.0 * u(B)
    inside = ((x - cx[:, None]) ** 2 + (y - cy[:, None]) ** 2) < (r ** 2)[:, None]
    for k, name in enumerate(kinds):
        rows = (kind == k)[:, None]
        if name == "melt":
            sp, sa = 0.02 + 0.2 * u(B, 1), 0.02 + 0.4 * u(B, 1)
            m = (inside & rows).to(x.dtype)
            x += m * sp * g(B, n)
            y += m * sp * g(B, n)
            t += m * sa * g(B, n)
        elif name == "rotate":
            phi = (0.05 + 0.4 * u(B)) * torch.sign(g(B))
            m = inside & rows
            ox, oy = x - cx[:, None], y - cy[:, None]
            c, s = phi.cos()[:, None], phi.sin()[:, None]
            x.copy_(torch.where(m, cx[:, None] + c * ox - s * oy, x))
            y.copy_(torch.where(m, cy[:, None] + s * ox + c * oy, y))
            t.copy_(torch.where(m, t + phi[:, None], t))
        elif name == "kick":
            m = rows.to(x.dtype)
            x += m * 0.01 * g(B, n)
            y += m * 0.01 * g(B, n)
            t += m * 0.01 * g(B, n)


def relax(x, y, t, L, I, J, steps: int, lr: float = 0.02):
    """Batched Adam on the overlap energy at fixed per-row sides L, with a cosine-decayed step."""
    x, y, t = (a.clone().requires_grad_(True) for a in (x, y, t))
    opt = torch.optim.Adam([x, y, t], lr=lr, betas=(0.9, 0.99))
    for k in range(steps):
        for gr in opt.param_groups:
            gr["lr"] = 1e-5 + 0.5 * (lr - 1e-5) * (1 + math.cos(math.pi * k / steps))
        opt.zero_grad(set_to_none=True)
        e, _ = batch_energy(x, y, t, L, I, J)
        e.sum().backward()
        opt.step()
    with torch.no_grad():
        e, worst = batch_energy(x, y, t, L, I, J)
    return x.detach(), y.detach(), t.detach(), e, worst


def finish(x, y, t, L, L0, maxiter=3000):
    """Float64: make a screened variant feasible at the smallest side in [L, L0) it reaches, then squeeze.

    Tries L, then points closer to the record side L0; returns (x, y, t, side) or None if the
    variant only fits at L0 or above.
    """
    for frac in (0.0, 0.5, 0.9, 0.99):
        Lt = L + frac * (L0 - L)
        k = Lt / L
        rx, ry, rt, _ = relax_lbfgs(x * k, y * k, t, Lt, MARGIN, maxiter)
        if feasible(rx, ry, rt, Lt, MARGIN):
            return squeeze(rx, ry, rt, Lt, rounds=20, step=(Lt - L) / Lt + 1e-6, maxiter=1500, method="lbfgs")
    return None


def polish_round(n, x0, y0, t0, L0, *, device, batch=256, steps=600, shrink=(2e-5, 2e-3),
                 top=4, pair_radius=3.0, gen=None, log=print):
    """One round from seed (x0, y0, t0 radians, L0). Returns the best (x, y, t, L) found, or None."""
    gen = gen or torch.Generator().manual_seed(0)
    dtype = torch.float64 if device.type == "cpu" else torch.float32
    I, J = (torch.as_tensor(a, device=device) for a in seed_pairs(x0, y0, pair_radius))
    rep = lambda a: torch.as_tensor(a, dtype=dtype, device=device).repeat(batch, 1)
    x, y, t = rep(x0), rep(y0), rep(t0)
    lo, hi = shrink
    f = torch.exp(math.log(lo) + (math.log(hi) - math.log(lo)) * torch.rand(batch, generator=gen)).to(device, dtype)
    L = torch.full((batch,), L0, dtype=dtype, device=device) * (1 - f)
    k = (L / L0)[:, None]
    x, y = x * k, y * k
    perturb(x, y, t, L, gen)
    x, y, t, e, worst = relax(x, y, t, L, I, J, steps)
    # Rank by overlap left per unit of shrink: a variant that nearly fits a much smaller box beats
    # one that fits a box barely below the record.
    rank = torch.argsort(worst.clamp(min=0) / f + 1e-12 * e)
    best = None
    for b in rank[:top].tolist():
        xb, yb, tb = (a[b].double().cpu().numpy() for a in (x, y, t))
        Lb = float(L[b])
        out = finish(xb, yb, tb, Lb, L0)
        if out is None:
            continue
        if best is None or out[3] < best[3]:
            best = out
    if best is not None:
        log(f"  n={n} round best {best[3]:.10f} (seed {L0:.10f}, d={L0 - best[3]:+.2e})")
    return best


def polish_n(n: int, *, minutes: float, device, batch: int, steps: int, root=None, seed: int = 0,
             log=print) -> dict:
    """Run rounds on n from the stored packing until the time budget is spent; offer improvements."""
    e = best_store.load(n, root)
    if e is None:
        return {"n": n, "skipped": "no stored packing"}
    x = np.array(e["x"], dtype=float)
    y = np.array(e["y"], dtype=float)
    t = np.array(e["angle"], dtype=float)
    L = float(e["side"])
    start_side = L
    gen = torch.Generator().manual_seed(seed * 100003 + n)
    deadline = time.time() + 60 * minutes
    rounds = wins = 0
    while time.time() < deadline:
        rounds += 1
        out = polish_round(n, x, y, t, L, device=device, batch=batch, steps=steps, gen=gen, log=log)
        if out is None or out[3] >= L - 1e-10:
            continue
        r = best_store.offer(n, out[0], out[1], np.rad2deg(out[2]), "gpu_polish", root=root)
        log(f"  n={n} offer {r['side']} -> {r['reason']}")
        if r["accepted"]:
            wins += 1
            x, y, t, L = out[0], out[1], out[2], float(r["side"])
    return {"n": n, "rounds": rounds, "wins": wins, "start": start_side, "end": L}
