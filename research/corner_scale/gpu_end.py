"""Batched GPU basin hopping for tilted-column end gadgets.

B copies of one gadget (trapezoid x >= 0, 0 <= y <= H, x cos phi - y sin phi <= c) evolve in parallel.
Each round every copy gets a random move (kick, local shake, rotate-to-0/-phi, remove+reinsert into a
hole, insert one square into a hole), is compacted by Adam on c + mu E with growing mu, repaired at
fixed c, and accepted if its waste e = m (c + s H / 2) - N drops. Squares are padded to Nmax with an
active mask so copies can carry different N. Candidates are re-checked in float64 on the CPU with
endgadget.Gadget (margin 2e-9) before being saved.

usage: python -m research.corner_scale.gpu_end r f minutes outdir [B] [seed]
"""
import json, math, os, sys, time
import numpy as np
import torch
from research.corner_scale.dpbuild import best_dp, squares
from research.corner_scale.endgadget import Gadget, hext

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
DT = torch.float32
MARGIN_GPU = 2e-6     # search margin: float32 slack so the float64 check (2e-9) passes
CHECK_MARGIN = 2e-9


def habs(a):
    return 0.5 * (a.cos().abs() + a.sin().abs())


class Batch:
    def __init__(self, r, f, phi, B, Nmax):
        self.r, self.f, self.phi = r, f, phi
        self.m, self.H = r + 1, r + f
        self.cs, self.sn = math.cos(phi), math.sin(phi)
        self.B, self.Nmax = B, Nmax
        iu = torch.triu_indices(Nmax, Nmax, 1, device=DEV)
        self.pi, self.pj = iu[0], iu[1]
        self.energy = torch.compile(self.energy)

    def energy(self, x, y, t, c, act, margin):
        """Per-copy overlap energy (B,), and per-copy worst pair depth / wall violation."""
        i, j = self.pi, self.pj
        dx = x[:, j] - x[:, i]; dy = y[:, j] - y[:, i]
        ci, si, cj, sj = t[:, i].cos(), t[:, i].sin(), t[:, j].cos(), t[:, j].sin()
        rad = 0.5 + habs(t[:, j] - t[:, i]) + margin
        p0 = (dx * ci + dy * si).abs(); p1 = (-dx * si + dy * ci).abs()
        p2 = (dx * cj + dy * sj).abs(); p3 = (-dx * sj + dy * cj).abs()
        pen = rad - torch.maximum(torch.maximum(p0, p1), torch.maximum(p2, p3))
        pm = act[:, i] & act[:, j]
        pen = torch.where(pm, pen, torch.full_like(pen, -1.0))
        hw = habs(t)
        w1 = hw + margin - x
        w2 = hw + margin - y
        w3 = y + hw + margin - self.H
        w4 = x * self.cs - y * self.sn + habs(t + self.phi) + margin - c[:, None]
        walls = torch.stack([w1, w2, w3, w4], 0)
        walls = torch.where(act[None], walls, torch.full_like(walls, -1.0))
        e = pen.clamp(min=0).pow(2).sum(1) + walls.clamp(min=0).pow(2).sum((0, 2))
        worst = torch.maximum(pen.max(1).values, walls.amax((0, 2)))
        return e, worst

    def creq(self, x, y, t, act):
        u = x * self.cs - y * self.sn + habs(t + self.phi)
        return torch.where(act, u, torch.full_like(u, -1e9)).max(1).values

    def waste(self, N, c):
        return self.m * (c + self.sn * self.H / 2) - N

    def compact(self, x, y, t, c, act, steps, mus=(1e1, 1e2, 1e3, 1e4, 1e5)):
        x, y, t, c = [v.clone().requires_grad_(True) for v in (x, y, t, c)]
        opt = torch.optim.Adam([x, y, t, c], lr=3e-3)
        for k, mu in enumerate(mus):
            lr = 3e-3 * (0.5 ** k)
            for g in opt.param_groups: g['lr'] = lr
            for _ in range(steps):
                opt.zero_grad()
                e, _ = self.energy(x, y, t, c, act, MARGIN_GPU)
                (c.sum() + mu * e.sum()).backward()
                opt.step()
        return x.detach(), y.detach(), t.detach(), c.detach()

    def repair(self, x, y, t, c, act, steps):
        """Fixed c: gradient descent on E alone, then set c to the exact requirement."""
        x, y, t = [v.clone().requires_grad_(True) for v in (x, y, t)]
        c = c + 1e-5
        opt = torch.optim.Adam([x, y, t], lr=2e-5)
        for _ in range(steps):
            opt.zero_grad()
            e, _ = self.energy(x, y, t, c, act, 2 * MARGIN_GPU)
            e.sum().backward()
            opt.step()
        with torch.no_grad():
            c2 = self.creq(x, y, t, act) + MARGIN_GPU
            _, worst = self.energy(x, y, t, c2, act, MARGIN_GPU)
        return x.detach(), y.detach(), t.detach(), c2, worst


def holes(G, x, y, act, c, K=256):
    """Per copy: the sampled trapezoid point farthest (Chebyshev) from every active center."""
    B = x.shape[0]
    yy = torch.rand(B, K, device=DEV, dtype=DT) * (G.H - 1.0) + 0.5
    xmax = (c[:, None] - 0.7 + yy * G.sn) / G.cs
    xx = 0.5 + torch.rand(B, K, device=DEV, dtype=DT) * (xmax - 0.5).clamp(min=0.05)
    d = torch.maximum((xx[:, :, None] - x[:, None, :]).abs(), (yy[:, :, None] - y[:, None, :]).abs())
    d = torch.where(act[:, None, :], d, torch.full_like(d, 1e3)).min(2).values
    k = d.argmax(1)
    ar = torch.arange(B, device=DEV)
    return xx[ar, k], yy[ar, k]


def perturb(G, x, y, t, c, act, gen):
    B, N = x.shape
    x, y, t, c, act = x.clone(), y.clone(), t.clone(), c.clone(), act.clone()
    ar = torch.arange(B, device=DEV)
    mv = torch.randint(0, 6, (B,), device=DEV, generator=gen)
    rnd = lambda *s: torch.randn(*s, device=DEV, dtype=DT, generator=gen)
    uni = lambda *s: torch.rand(*s, device=DEV, dtype=DT, generator=gen)
    # 0: global kick
    sig = torch.tensor([0.01, 0.03, 0.08], device=DEV, dtype=DT)[torch.randint(0, 3, (B,), device=DEV, generator=gen)]
    m0 = (mv == 0)[:, None] & act
    x = x + m0 * rnd(B, N) * sig[:, None]; y = y + m0 * rnd(B, N) * sig[:, None]; t = t + m0 * rnd(B, N) * sig[:, None] * 0.5
    # 1: local shake around a random active square
    pick = torch.multinomial(act.to(DT), 1, generator=gen)[:, 0]
    px, py = x[ar, pick], y[ar, pick]
    near = (((x - px[:, None]) ** 2 + (y - py[:, None]) ** 2) < 2.5 ** 2) & act & (mv == 1)[:, None]
    x = x + near * rnd(B, N) * 0.25; y = y + near * rnd(B, N) * 0.25; t = t + near * rnd(B, N) * 0.3
    # 2: snap 1-3 squares to angle 0 or -phi
    for _ in range(3):
        k = torch.multinomial(act.to(DT), 1, generator=gen)[:, 0]
        ang = torch.where(uni(B) < 0.5, torch.zeros(B, device=DEV, dtype=DT), torch.full((B,), -G.phi, device=DEV, dtype=DT))
        sel = (mv == 2) & (uni(B) < 0.7)
        t[ar[sel], k[sel]] = ang[sel]
        x[ar[sel], k[sel]] += rnd(int(sel.sum())) * 0.1
    # 3: remove one square and reinsert it in the largest hole; 4: insert a new square there
    hx, hy = holes(G, x, y, act, c)
    k = torch.multinomial(act.to(DT), 1, generator=gen)[:, 0]
    sel = mv == 3
    x[ar[sel], k[sel]] = hx[sel]; y[ar[sel], k[sel]] = hy[sel]
    t[ar[sel], k[sel]] = torch.where(uni(int(sel.sum())) < 0.5, 0.0, -G.phi).to(DT)
    free = (~act).to(DT)
    has_free = free.sum(1) > 0
    sel = (mv == 4) & has_free
    k = torch.multinomial(free + 1e-12, 1, generator=gen)[:, 0]
    x[ar[sel], k[sel]] = hx[sel]; y[ar[sel], k[sel]] = hy[sel]
    t[ar[sel], k[sel]] = torch.where(uni(int(sel.sum())) < 0.5, 0.0, -G.phi).to(DT)
    act[ar[sel], k[sel]] = True
    c = c + sel.to(DT) * (1.0 / G.m)
    # 5: remove one square and reinsert two (net +1) via a region shake
    sel = (mv == 5) & has_free
    k2 = torch.multinomial(act.to(DT), 1, generator=gen)[:, 0]
    x[ar[sel], k[sel]] = x[ar[sel], k2[sel]] + 0.3 * rnd(int(sel.sum()))
    y[ar[sel], k[sel]] = y[ar[sel], k2[sel]] + 0.3 * rnd(int(sel.sum()))
    t[ar[sel], k[sel]] = t[ar[sel], k2[sel]]
    act[ar[sel], k[sel]] = True
    c = c + sel.to(DT) * (1.5 / G.m)
    # small slack so compaction can rearrange
    c = c + 0.02 + 0.05 * uni(B)
    return x, y, t, c, act


def cpu_check(r, f, phi, x, y, t, c):
    G = Gadget(r, f, phi)
    x, y, t = [np.asarray(v, dtype=np.float64) for v in (x, y, t)]
    creq = float((x * G.cs - y * G.sn + hext(t, G.psi)).max()) + CHECK_MARGIN
    c = max(c, creq)
    _, *_, worst = G.energy(x, y, t, c, CHECK_MARGIN)
    if worst > 0 or not G.feasible(x, y, t, c):
        # one float64 relax at fixed c, as a last resort
        x, y, t = G.relax(x, y, t, c + 1e-7)
        c = float((x * G.cs - y * G.sn + hext(t, G.psi)).max()) + CHECK_MARGIN
        _, *_, worst = G.energy(x, y, t, c, CHECK_MARGIN)
        if worst > 0 or not G.feasible(x, y, t, c):
            return None
    return x, y, t, c, G.waste(len(x), c)


def run(r, f, minutes, outdir, B=512, seed=0, extra=8, csteps=120, rsteps=200):
    torch.manual_seed(seed)
    gen = torch.Generator(device=DEV); gen.manual_seed(seed)
    w0, S, J, N0, kind = best_dp(r, f)
    sq = squares(S, J)
    phi = S.phi
    G = Batch(r, f, phi, B, len(sq) + extra)
    Nmax = G.Nmax
    x = torch.zeros(B, Nmax, device=DEV, dtype=DT); y = torch.zeros_like(x); t = torch.zeros_like(x)
    act = torch.zeros(B, Nmax, dtype=torch.bool, device=DEV)
    n = len(sq)
    x[:, :n] = torch.tensor(sq[:, 0], dtype=DT, device=DEV)
    y[:, :n] = torch.tensor(sq[:, 1], dtype=DT, device=DEV)
    t[:, :n] = torch.tensor(sq[:, 2], dtype=DT, device=DEV)
    act[:, :n] = True
    c = torch.full((B,), S.c, dtype=DT, device=DEV)
    # parked squares sit away from everything; masked out of the energy anyway
    x[:, n:] = -50.0; y[:, n:] = -50.0
    bx, by, bt, bc, bact = x, y, t, c + 1e-4, act
    bw = G.waste(bact.sum(1).to(DT), bc)
    best_saved = (w0, None)
    print(f'r={r} f={f} phi={phi:.4f} DP waste {w0:.4f} /row {w0 / G.H:.4f} N={N0} B={B}', flush=True)
    t0 = time.time(); rnd = 0
    out = os.path.join(outdir, f'end_r{r}_f{f:.2f}.json')
    while time.time() - t0 < minutes * 60:
        rnd += 1
        x, y, t, c, act = perturb(G, bx, by, bt, bc, bact, gen)
        x = torch.where(act, x, torch.full_like(x, -50.0)); y = torch.where(act, y, torch.full_like(y, -50.0))
        x, y, t, c = G.compact(x, y, t, c, act, csteps)
        x, y, t, c2, worst = G.repair(x, y, t, c, act, rsteps)
        ok = worst <= 0
        w = G.waste(act.sum(1).to(DT), c2)
        # accept improvements; occasionally accept small uphill steps to keep moving
        temp = 0.02
        acc = ok & ((w < bw - 1e-6) | (torch.rand(B, device=DEV, generator=gen) < torch.exp(-(w - bw).clamp(min=0) / temp) * 0.1))
        bx = torch.where(acc[:, None], x, bx); by = torch.where(acc[:, None], y, by)
        bt = torch.where(acc[:, None], t, bt); bact = torch.where(acc[:, None], act, bact)
        bc = torch.where(acc, c2, bc); bw = torch.where(acc, w, bw)
        # population step: worst quarter restarts from the best copies
        if rnd % 10 == 0:
            order = torch.argsort(bw)
            q = B // 4
            src = order[torch.randint(0, q, (q,), device=DEV, generator=gen)]
            dst = order[-q:]
            bx[dst], by[dst], bt[dst], bc[dst], bact[dst], bw[dst] = bx[src], by[src], bt[src], bc[src], bact[src], bw[src]
        k = int(torch.argmin(bw))
        if float(bw[k]) < best_saved[0] - 1e-5:
            a = bact[k].cpu().numpy()
            res = cpu_check(r, f, phi, bx[k].cpu().numpy()[a], by[k].cpu().numpy()[a], bt[k].cpu().numpy()[a], float(bc[k]))
            if res is not None and res[4] < best_saved[0] - 1e-5:
                xs, ys, ts, cs_, ws = res
                best_saved = (ws, res)
                json.dump({'r': r, 'f': f, 'phi': phi, 'N': len(xs), 'c': cs_, 'waste': ws, 'dp_waste': w0,
                           'x': xs.tolist(), 'y': ys.tolist(), 't': ts.tolist()}, open(out, 'w'))
                print(f'  rnd {rnd} {time.time() - t0:.0f}s N={len(xs)} waste {ws:.4f} /row {ws / G.H:.4f} (saved)', flush=True)
        if rnd % 20 == 0:
            print(f'  rnd {rnd} {time.time() - t0:.0f}s best {float(bw.min()):.4f} median {float(bw.median()):.4f} '
                  f'acc {float(acc.to(DT).mean()):.2f} ok {float(ok.to(DT).mean()):.2f}', flush=True)
    ws = best_saved[0]
    print(f'DONE r={r} f={f} dp {w0 / G.H:.4f} best {ws / G.H:.4f} /row', flush=True)
    return w0, ws


if __name__ == '__main__':
    r, f, minutes, outdir = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
    B = int(sys.argv[5]) if len(sys.argv) > 5 else 512
    seed = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    os.makedirs(outdir, exist_ok=True)
    run(r, f, minutes, outdir, B, seed)
