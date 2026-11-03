"""Batched GPU basin hopping for tilted-column end gadgets.

B copies of one gadget (trapezoid x >= 0, 0 <= y <= H, x cos phi - y sin phi <= c) evolve in parallel.
Each round every copy gets a random move (kick, local shake, rotate-to-0/-phi, remove+reinsert into a
hole, insert one square into a hole) plus some slack in c, is relaxed by batched FIRE at fixed c,
then squeezed (per-copy line search on c, FIRE at each trial), and accepted by annealed Metropolis on
the waste e = m (c + s H / 2) - N. Squares are padded to Nmax with an active mask so copies can carry
different N. Every new global best is re-checked in float64 on the CPU with endgadget.Gadget.

A full column of r+1 squares at the exact band tilt fills H with zero slack, so no positive margin is
feasible; the search tilts by PHI_SLACK more (columns get 1e-4 of room, waste changes by ~1e-3, ~1e-4 per row).

usage: python -m research.corner_scale.gpu_end r f minutes outdir [B] [seed]
"""
import json, math, os, sys, time
import numpy as np
import torch
from research.corner_scale.dpbuild import best_dp, squares
from research.corner_scale.endgadget import Gadget, hext

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
DT = torch.float32
MARGIN_GPU = 1e-6     # float32 search margin, well above the CPU check margin
CHECK_MARGIN = 2e-9
COL_SLACK = 1e-4      # height slack given to a full column by tilting a bit more


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
        # angles stay at their discrete values (0 or -phi; changed only by moves): a full column is
        # tight in height, and FIRE cannot undo a small tilt error of a whole column
        self.zmask = torch.tensor([1.0, 1.0, 0.0], device=DEV, dtype=DT)[:, None, None]

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

    def fire(self, x, y, t, c, act, steps, margin=2 * MARGIN_GPU, dt0=0.02, dtmax=0.2):
        """Batched FIRE on E at fixed per-copy c (one velocity / timestep state per copy)."""
        B = x.shape[0]
        z = torch.stack([x, y, t], 0).detach().clone()
        v = torch.zeros_like(z)
        dt = torch.full((B,), dt0, device=DEV, dtype=DT); alpha = torch.full((B,), 0.1, device=DEV, dtype=DT)
        npos = torch.zeros(B, device=DEV, dtype=torch.long)
        actf = act.to(DT)[None] * self.zmask
        for _ in range(steps):
            zz = z.requires_grad_(True)
            e, _ = self.energy(zz[0], zz[1], zz[2], c, act, margin)
            gz, = torch.autograd.grad(e.sum(), zz)
            z = z.detach(); F = -gz * actf
            P = (F * v).sum((0, 2))
            vn = v.pow(2).sum((0, 2)).sqrt(); fn = F.pow(2).sum((0, 2)).sqrt() + 1e-300
            pos = P > 0
            a = alpha[None, :, None]
            v = torch.where(pos[None, :, None], (1 - a) * v + a * (vn / fn)[None, :, None] * F, torch.zeros_like(v))
            npos = torch.where(pos, npos + 1, torch.zeros_like(npos))
            grow = pos & (npos > 5)
            dt = torch.where(grow, (dt * 1.1).clamp(max=dtmax), torch.where(pos, dt, dt * 0.5))
            alpha = torch.where(grow, alpha * 0.99, torch.where(pos, alpha, torch.full_like(alpha, 0.1)))
            v = v + dt[None, :, None] * F
            z = z + dt[None, :, None] * v
        with torch.no_grad():
            _, worst = self.energy(z[0], z[1], z[2], c, act, MARGIN_GPU)
        return z[0], z[1], z[2], worst

    def squeeze(self, x, y, t, c, act, ok, rounds, steps, step0=3e-3):
        """Per-copy line search on c: try c - step, FIRE, keep if feasible; grow/shrink step."""
        step = torch.full_like(c, step0)
        for _ in range(rounds):
            ct = c - step
            nx, ny, nt, worst = self.fire(x, y, t, ct, act, steps)
            good = ok & (worst <= 0)
            x = torch.where(good[:, None], nx, x); y = torch.where(good[:, None], ny, y)
            t = torch.where(good[:, None], nt, t)
            c = torch.where(good, torch.minimum(ct, self.creq(nx, ny, nt, act) + MARGIN_GPU), c)
            step = torch.where(good, step * 1.5, step * 0.4)
        return x, y, t, c


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
    x = x + m0 * rnd(B, N) * sig[:, None]; y = y + m0 * rnd(B, N) * sig[:, None]
    # 1: local shake around a random active square
    pick = torch.multinomial(act.to(DT), 1, generator=gen)[:, 0]
    px, py = x[ar, pick], y[ar, pick]
    near = (((x - px[:, None]) ** 2 + (y - py[:, None]) ** 2) < 2.5 ** 2) & act & (mv == 1)[:, None]
    x = x + near * rnd(B, N) * 0.25; y = y + near * rnd(B, N) * 0.25
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
    c = c + 0.005 + 0.03 * uni(B)
    return x, y, t, c, act


def cpu_check(r, f, phi, x, y, t, c):
    """Exact float64 check with endgadget.Gadget; c is reset to the smallest feasible face."""
    G = Gadget(r, f, phi)
    x, y, t = [np.asarray(v, dtype=np.float64) for v in (x, y, t)]
    c = float((x * G.cs - y * G.sn + hext(t, G.psi)).max()) + CHECK_MARGIN
    _, *_, worst = G.energy(x, y, t, c, CHECK_MARGIN)
    if worst > 0 or not G.feasible(x, y, t, c):
        return None
    return x, y, t, c, G.waste(len(x), c)


def tilt(r, f, slack):
    from scipy.optimize import brentq
    return brentq(lambda p: (r + 1) * math.cos(p) + math.sin(p) - (r + f - slack), 1e-9, 1.4)


def loose_dp(r, f, na=41, noff=9, room=3e-5):
    """dpbuild.best_dp, but row stacks and column segments must leave `room` of height."""
    import research.corner_scale.seqend as seqend
    import research.corner_scale.dpbuild as dpbuild
    from research.corner_scale.dpend import col_fit
    from research.corner_scale.seqend import Seq
    seqend.col_fit = dpbuild.col_fit = lambda a, y0, y1, cs, sn, m: col_fit(a, y0, y1 - room, cs, sn, m)
    try:
        phi = seqend.phi_for(r, f); hs = (r + f) * math.sin(phi)
        Jmax = int(math.ceil(hs)) + 6
        best = None
        for a0 in np.linspace(-hs - 1.5, -hs - 0.5, na):
            for off in np.linspace(0, f - room, noff):
                S = Seq(r, f, a0, off, Jmax)
                for kind in ('dec', 'inc'):
                    J, N = dpbuild.dp_seq(S, kind)
                    w = S.waste(N)
                    if best is None or w < best[0]:
                        best = (w, S, J)
    finally:
        seqend.col_fit = dpbuild.col_fit = col_fit
    return best[1], best[2]


def seed_run(r, f, phi, S, J, B, extra):
    """Relax and squeeze the DP gadget on the GPU; return the best copy broadcast to B, or None."""
    sq = squares(S, J)
    G = Batch(r, f, phi, B, len(sq) + extra)
    Nmax, n = G.Nmax, len(sq)
    x = torch.full((B, Nmax), -50.0, device=DEV, dtype=DT); y = x.clone(); t = torch.zeros_like(x)
    act = torch.zeros(B, Nmax, dtype=torch.bool, device=DEV)
    x[:, :n] = torch.tensor(sq[:, 0], dtype=DT, device=DEV)
    y[:, :n] = torch.tensor(sq[:, 1], dtype=DT, device=DEV)
    t[:, :n] = torch.tensor(sq[:, 2], dtype=DT, device=DEV)
    act[:, :n] = True
    # seed: relax the DP gadget at the slightly larger tilt, then squeeze it
    c = torch.full((B,), S.c + 1e-3, dtype=DT, device=DEV)
    x, y, t, worst = G.fire(x, y, t, c, act, 1000)
    ok = worst <= 0
    c = torch.where(ok, G.creq(x, y, t, act) + MARGIN_GPU, c)
    x, y, t, c = G.squeeze(x, y, t, c, act, ok, 12, 200)
    k = int(torch.argmin(torch.where(ok, c, torch.full_like(c, 1e9))))
    if not bool(ok[k]):
        return None
    x, y, t, c = [v[k:k + 1].expand(B, *v.shape[1:]).clone() for v in (x, y, t, c)]
    act = act[k:k + 1].expand(B, Nmax).clone()
    return G, x, y, t, c, act


def run(r, f, minutes, outdir, B=512, seed=0, extra=8, rsteps=300, sq_rounds=8, sq_steps=150,
        T0=0.15, T1=0.01):
    torch.manual_seed(seed)
    gen = torch.Generator(device=DEV); gen.manual_seed(seed)
    import research.corner_scale.seqend as seqend
    phi = tilt(r, f, COL_SLACK)
    seqend.phi_for = lambda r_, f_: tilt(r_, f_, COL_SLACK)   # build the DP seed at the search tilt
    w0, S, J, N0, kind = best_dp(r, f)
    w = seed_run(r, f, phi, S, J, B, extra)
    if w is None:
        # exact fits elsewhere (rows stacked off..H when off = f, a truncated column wall to ceiling)
        # are also infeasible with a margin: rebuild the seed with a little room for both
        w = seed_run(r, f, phi, *loose_dp(r, f), B, extra)
    if w is None:
        print(f'r={r} f={f}: DP seed not feasible on GPU', flush=True); return w0, None
    G, x, y, t, c, act = w
    Nmax = G.Nmax
    out = os.path.join(outdir, f'end_r{r}_f{f:.2f}.json')
    cur = (x, y, t, c, act); cw = G.waste(act.sum(1).to(DT), c)
    a = act[0].cpu().numpy()
    res = cpu_check(r, f, phi, x[0].cpu().numpy()[a], y[0].cpu().numpy()[a], t[0].cpu().numpy()[a], float(c[0]))
    seed_w = res[4] if res else float('inf')
    best_w, saved = seed_w, res
    print(f'r={r} f={f} phi={S.phi:.4f} DP waste {w0:.4f} /row {w0 / G.H:.4f} N={N0}; '
          f'GPU seed {seed_w:.4f} /row {seed_w / G.H:.4f} (cpu ok={res is not None}) B={B}', flush=True)
    t0 = time.time(); rnd = 0
    while time.time() - t0 < minutes * 60:
        rnd += 1
        frac = min(1.0, (time.time() - t0) / (minutes * 60))
        T = T0 * (T1 / T0) ** frac
        px, py, pt, pc, pact = perturb(G, *cur, gen)
        px = torch.where(pact, px, torch.full_like(px, -50.0)); py = torch.where(pact, py, torch.full_like(py, -50.0))
        px, py, pt, worst = G.fire(px, py, pt, pc, pact, rsteps)
        ok = worst <= 0
        pc = torch.where(ok, G.creq(px, py, pt, pact) + MARGIN_GPU, pc)
        px, py, pt, pc = G.squeeze(px, py, pt, pc, pact, ok, sq_rounds, sq_steps)
        w = G.waste(pact.sum(1).to(DT), pc)
        u = torch.rand(B, device=DEV, dtype=DT, generator=gen)
        acc = ok & ((w < cw) | (u < torch.exp(-(w - cw) / T)))
        A = acc[:, None]
        cur = (torch.where(A, px, cur[0]), torch.where(A, py, cur[1]), torch.where(A, pt, cur[2]),
               torch.where(acc, pc, cur[3]), torch.where(A, pact, cur[4]))
        cw = torch.where(acc, w, cw)
        # population step: worst quarter restarts from the best quarter
        if rnd % 10 == 0:
            order = torch.argsort(cw); q = B // 4
            src = order[torch.randint(0, q, (q,), device=DEV, generator=gen)]; dst = order[-q:]
            cur = tuple(v.clone() for v in cur)
            for v in cur: v[dst] = v[src]
            cw = cw.clone(); cw[dst] = cw[src]
        # global best (checked on CPU) across this round's feasible candidates
        wk = torch.where(ok, w, torch.full_like(w, 1e9))
        for k in torch.argsort(wk)[:4].tolist():
            if float(wk[k]) >= best_w - 1e-6: break
            a = pact[k].cpu().numpy()
            res = cpu_check(r, f, phi, px[k].cpu().numpy()[a], py[k].cpu().numpy()[a], pt[k].cpu().numpy()[a], float(pc[k]))
            if res is not None and res[4] < best_w - 1e-6:
                best_w, saved = res[4], res
                xs, ys, ts, cs_, ws = res
                json.dump({'r': r, 'f': f, 'phi': phi, 'N': len(xs), 'c': cs_, 'waste': ws, 'dp_waste': w0,
                           'x': xs.tolist(), 'y': ys.tolist(), 't': ts.tolist()}, open(out, 'w'))
                print(f'  rnd {rnd} {time.time() - t0:.0f}s N={len(xs)} waste {ws:.4f} /row {ws / G.H:.4f} (saved)', flush=True)
                break
        if rnd % 25 == 0:
            print(f'  rnd {rnd} {time.time() - t0:.0f}s T={T:.3f} pop best {float(cw.min()):.4f} median {float(cw.median()):.4f} '
                  f'ok {float(ok.to(DT).mean()):.2f} acc {float(acc.to(DT).mean()):.2f}', flush=True)
    print(f'DONE r={r} f={f} dp {w0 / G.H:.4f} seed {seed_w / G.H:.4f} best {best_w / G.H:.4f} /row', flush=True)
    return w0, best_w


if __name__ == '__main__':
    r, f, minutes, outdir = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
    B = int(sys.argv[5]) if len(sys.argv) > 5 else 512
    seed = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    os.makedirs(outdir, exist_ok=True)
    run(r, f, minutes, outdir, B, seed)
