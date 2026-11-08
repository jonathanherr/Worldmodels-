"""Q-learning over macro moves in a fixed, too-small box ("squeeze-feasibility MDP").

The box is fixed at a target side L* below our record side (e.g. the Kaggle per-n leader's side).
The record packing is scaled into it and relaxed; it keeps a positive overlap energy E because the
record basin is rigid. The agent's job is to drive E to exactly zero (= a feasible packing at L*).

State      : the relaxed packing (x, y, t) at L*.
Actions    : macro moves on one of the K most-overlapping squares, each followed by a FIRE relax at
             L*: kick (8 directions), turn (+-), relocate to the biggest hole, rigid turn of its
             6-neighbour cluster (+-), swap with a neighbour of different tilt.
Reward     : log(E_before) - log(E_after); +10 and stop when E hits 0.
Q(s, a)    : linear in hand-made features of (state, move) -- move-type one-hots crossed with the
             square's overlap share, local tilt mix, wall distance, global log E, step fraction.

Policies compared at equal environment-step budget: q (epsilon-greedy Q-learning), random,
greedy (try m candidates per step, keep the best immediate reward: the classic shaking baseline).

  python qpack.py CSV N --target L* --policy q --minutes 20 --seed 0 --out run.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from wm.pack_opt import MARGIN, energy_grad, feasible, relax_fire  # noqa: E402

MOVES = ["kick", "turn", "relocate", "cluster", "swap"]
K = 12          # candidate squares (most overlapping)
RELAX_IT = 3000


def load_csv_n(path, n):
    xs, ys, ts = [], [], []
    with open(path) as f:
        next(f)
        for line in f:
            p = line.rstrip().split(",")
            if int(p[1]) == n:
                xs.append(float(p[3].lstrip("s"))); ys.append(float(p[4].lstrip("s"))); ts.append(float(p[5].lstrip("s")))
    return np.array(xs), np.array(ys), np.array(ts)


def half_extent(t):
    return 0.5 * (np.abs(np.cos(t)) + np.abs(np.sin(t)))


def per_square_overlap(x, y, t, L):
    """Overlap energy attributed to each square (pair depth^2 split, plus wall penetration^2)."""
    n = len(x)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d2 = dx * dx + dy * dy
    e = np.zeros(n)
    I, J = np.nonzero(np.triu(d2 < 2.0, 1))
    for i, j in zip(I, J):
        # exact SAT depth for one pair
        best = 1e9
        for ang in (t[i], t[i] + math.pi / 2, t[j], t[j] + math.pi / 2):
            ax, ay = math.cos(ang), math.sin(ang)
            proj = abs(dx[i, j] * ax + dy[i, j] * ay)
            ri = 0.5 * (abs(math.cos(t[i] - ang)) + abs(math.sin(t[i] - ang)))
            rj = 0.5 * (abs(math.cos(t[j] - ang)) + abs(math.sin(t[j] - ang)))
            best = min(best, ri + rj - proj)
        if best > 0:
            e[i] += best * best / 2; e[j] += best * best / 2
    w = half_extent(t)
    for arr in (w - x, x + w - L, w - y, y + w - L):
        e += np.maximum(arr, 0) ** 2
    return e


class Env:
    def __init__(self, x, y, t, L0, Lt, horizon, rng):
        self.x0, self.y0, self.t0 = x * Lt / L0, y * Lt / L0, t.copy()
        self.L = Lt
        self.H = horizon
        self.rng = rng
        x, y, t, e = relax_fire(self.x0, self.y0, self.t0, Lt, MARGIN, 20000)
        self.start = (x, y, t, e)
        self.reset()

    def reset(self, state=None):
        x, y, t, e = state or self.start
        self.x, self.y, self.t, self.E = x.copy(), y.copy(), t.copy(), e
        self.step_i = 0
        self._cache()
        return self

    def _cache(self):
        self.es = per_square_overlap(self.x, self.y, self.t, self.L)
        self.cand = np.argsort(-self.es)[:K]

    # ---------- features ----------
    def square_feats(self, i):
        x, y, t, L = self.x, self.y, self.t, self.L
        d2 = (x - x[i]) ** 2 + (y - y[i]) ** 2
        nb = np.nonzero((d2 < 2.3) & (d2 > 0))[0]
        tilt = lambda a: min(a % (math.pi / 2), math.pi / 2 - a % (math.pi / 2))
        ti = tilt(t[i])
        mix = np.mean([abs(tilt(t[j]) - ti) > 0.05 for j in nb]) if len(nb) else 0.0
        wall = min(x[i], y[i], L - x[i], L - y[i])
        share = self.es[i] / (self.es.sum() + 1e-300)
        return np.array([1.0, share, math.sqrt(share), mix, ti / (math.pi / 4), min(wall, 3) / 3,
                         len(nb) / 8, math.log10(self.E + 1e-30) / 10, self.step_i / self.H])

    def actions(self):
        """All (move, square, arg) actions available now."""
        acts = []
        for i in self.cand:
            for d in range(8):
                acts.append(("kick", int(i), d))
            acts += [("turn", int(i), 1), ("turn", int(i), -1), ("relocate", int(i), 0),
                     ("cluster", int(i), 1), ("cluster", int(i), -1), ("swap", int(i), 0)]
        return acts

    def feats(self, acts):
        sq = {}
        F = np.zeros((len(acts), len(MOVES) * 9 + 8))
        for r, (m, i, a) in enumerate(acts):
            if i not in sq:
                sq[i] = self.square_feats(i)
            k = MOVES.index(m)
            F[r, k * 9:(k + 1) * 9] = sq[i]
            if m == "kick":  # direction relative to the overlap gradient
                F[r, len(MOVES) * 9 + 0] = self._kick_align(i, a)
            F[r, len(MOVES) * 9 + 1 + min(k, 4)] = 1.0
        return F

    def _kick_align(self, i, d):
        _, gx, gy, _, _, _ = self._grad
        g = np.array([gx[i], gy[i]])
        u = np.array([math.cos(d * math.pi / 4), math.sin(d * math.pi / 4)])
        return float(-(g @ u) / (np.linalg.norm(g) + 1e-30))

    def prep(self):
        self._grad = energy_grad(self.x, self.y, self.t, self.L, MARGIN)

    # ---------- dynamics ----------
    def apply(self, act):
        m, i, a = act
        x, y, t, L, rng = self.x.copy(), self.y.copy(), self.t.copy(), self.L, self.rng
        if m == "kick":
            s = math.exp(rng.uniform(math.log(0.005), math.log(0.2)))
            x[i] += s * math.cos(a * math.pi / 4); y[i] += s * math.sin(a * math.pi / 4)
        elif m == "turn":
            t[i] += a * math.exp(rng.uniform(math.log(0.005), math.log(0.3)))
        elif m == "relocate":
            # biggest hole: grid point farthest from all centres (rough), then drop square there
            g = np.linspace(0.5, L - 0.5, 40)
            GX, GY = np.meshgrid(g, g)
            P = np.stack([GX.ravel(), GY.ravel()], 1)
            mask = np.ones(len(x), bool); mask[i] = False
            d = np.min((P[:, None, 0] - x[None, mask]) ** 2 + (P[:, None, 1] - y[None, mask]) ** 2, 1)
            j = int(np.argmax(d)); x[i], y[i] = P[j]
            t[i] = rng.choice([0.0, math.pi / 4, math.atan(1 / 3)])
        elif m == "cluster":
            d2 = (x - x[i]) ** 2 + (y - y[i]) ** 2
            idx = np.argsort(d2)[:7]
            cx, cy = x[idx].mean(), y[idx].mean()
            th = a * math.exp(rng.uniform(math.log(0.003), math.log(0.12)))
            c, s = math.cos(th), math.sin(th)
            X, Y = x[idx] - cx, y[idx] - cy
            x[idx], y[idx] = cx + c * X - s * Y, cy + s * X + c * Y
            t[idx] += th
        elif m == "swap":
            d2 = (x - x[i]) ** 2 + (y - y[i]) ** 2
            tl = lambda v: min(v % (math.pi / 2), math.pi / 2 - v % (math.pi / 2))
            nb = [j for j in np.argsort(d2)[1:9] if abs(tl(t[j]) - tl(t[i])) > 0.05]
            if not nb:
                nb = list(np.argsort(d2)[1:4])
            j = int(rng.choice(nb))
            x[i], x[j] = x[j], x[i]; y[i], y[j] = y[j], y[i]
        x, y, t, e = relax_fire(x, y, t, L, MARGIN, RELAX_IT)
        r = math.log(self.E + 1e-30) - math.log(e + 1e-30)
        done = e == 0.0
        if done:
            r += 10.0
        self.x, self.y, self.t, self.E = x, y, t, e
        self.step_i += 1
        self._cache()
        return r, done or self.step_i >= self.H


def run(a):
    rng = np.random.default_rng(a.seed)
    x, y, t = load_csv_n(a.csv, a.n)
    from wm.pack_opt import squeeze
    L0 = float(max((x + half_extent(t)).max(), (y + half_extent(t)).max()))
    Lt = a.target
    env = Env(x, y, t, L0, Lt, a.horizon, rng)
    E0 = env.start[3]
    out = open(a.out, "w")
    out.write(json.dumps(dict(start=True, n=a.n, L0=L0, Lt=Lt, E0=E0, policy=a.policy, seed=a.seed)) + "\n")
    nf = len(MOVES) * 9 + 8
    w = np.zeros(nf)
    best = (E0, None)
    steps, ep = 0, 0
    t_end = time.time() + 60 * a.minutes
    eps = a.eps
    while time.time() < t_end and steps < a.max_steps:
        env.reset(best[1] if (a.restart_best and best[1] is not None and rng.random() < 0.5) else None)
        ep += 1
        ep_ret, ep_min = 0.0, env.E
        done = False
        while not done and time.time() < t_end:
            env.prep()
            acts = env.actions()
            F = env.feats(acts)
            if a.policy == "random":
                k = int(rng.integers(len(acts)))
            elif a.policy == "greedy":
                # try m candidates, keep best immediate reward (each try costs one env step)
                tries = rng.choice(len(acts), size=min(a.m, len(acts)), replace=False)
                snap = (env.x, env.y, env.t, env.E, env.step_i, env.es, env.cand)
                res = []
                for k in tries:
                    r, _ = env.apply(acts[k]); steps += 1
                    res.append((r, k, (env.x, env.y, env.t, env.E, env.es, env.cand)))
                    env.x, env.y, env.t, env.E, env.step_i, env.es, env.cand = snap
                r, k, st = max(res, key=lambda z: z[0])
                env.x, env.y, env.t, env.E, env.es, env.cand = st
                env.step_i += 1
                done = env.E == 0.0 or env.step_i >= env.H
                ep_ret += r; ep_min = min(ep_min, env.E)
                if env.E < best[0]:
                    best = (env.E, (env.x.copy(), env.y.copy(), env.t.copy(), env.E))
                continue
            else:
                q = F @ w
                k = int(rng.integers(len(acts))) if rng.random() < eps else int(np.argmax(q))
            r, done = env.apply(acts[k]); steps += 1
            ep_ret += r; ep_min = min(ep_min, env.E)
            if env.E < best[0]:
                best = (env.E, (env.x.copy(), env.y.copy(), env.t.copy(), env.E))
            if a.policy == "q":
                target = r
                if not done:
                    env.prep()
                    target += a.gamma * float(np.max(env.feats(env.actions()) @ w))
                td = target - F[k] @ w
                w += a.lr * td * F[k] / (1.0 + F[k] @ F[k])
        eps = max(0.05, eps * a.eps_decay)
        out.write(json.dumps(dict(ep=ep, steps=steps, ret=ep_ret, ep_min=ep_min, best=best[0],
                                  E_end=env.E)) + "\n"); out.flush()
        if best[0] == 0.0:
            break
    res = dict(summary=True, n=a.n, policy=a.policy, seed=a.seed, steps=steps, episodes=ep, E0=E0,
               best=best[0], ratio=best[0] / E0, w=w.tolist())
    if best[0] == 0.0:
        bx, by, bt, _ = best[1]
        assert feasible(bx, by, bt, Lt)
        sx, sy, st, sL = squeeze(bx, by, bt, Lt, rounds=40, step=1e-4)
        res["feasible_L"] = sL
        np.savez(a.out.replace(".jsonl", ".npz"), x=sx, y=sy, t=st, L=sL)
    elif best[1] is not None:
        bx, by, bt, _ = best[1]
        np.savez(a.out.replace(".jsonl", "_best.npz"), x=bx, y=by, t=bt, L=Lt, E=best[0])
    out.write(json.dumps(res) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "w"}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv"); ap.add_argument("n", type=int)
    ap.add_argument("--target", type=float, required=True)
    ap.add_argument("--policy", choices=["q", "random", "greedy"], default="q")
    ap.add_argument("--minutes", type=float, default=10); ap.add_argument("--max-steps", type=int, default=10**9)
    ap.add_argument("--horizon", type=int, default=25); ap.add_argument("--m", type=int, default=6)
    ap.add_argument("--gamma", type=float, default=0.9); ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--eps", type=float, default=0.5); ap.add_argument("--eps-decay", type=float, default=0.97)
    ap.add_argument("--restart-best", action="store_true")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", required=True)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
