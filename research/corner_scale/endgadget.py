"""Optimize the end of a tilted-column band: squares in the trapezoid
x >= 0, 0 <= y <= H, x cos(phi) - y sin(phi) <= c  (c = left face of the first full column).
End waste e = m (c + s H / 2) - N, where m = r + 1 squares per column, s = sin(phi).
"""
import math, sys
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0, '/home/claude/worldmodels-')
from wm.pack_opt import energy_grad, MARGIN

BIG = 1e6

def hext(t, psi):
    a = t - psi
    return 0.5 * (np.abs(np.cos(a)) + np.abs(np.sin(a)))

def dhext(t, psi):
    a = t - psi
    return 0.5 * (-np.sign(np.cos(a)) * np.sin(a) + np.sign(np.sin(a)) * np.cos(a))

class Gadget:
    def __init__(self, r, f, phi):
        self.r, self.f, self.phi = r, f, phi
        self.m = r + 1
        self.H = r + f
        self.cs, self.sn = math.cos(phi), math.sin(phi)
        self.psi = -phi  # direction of u = (cos phi, -sin phi)

    def energy(self, x, y, t, c, margin=MARGIN):
        e, gx, gy, gt, _, worst = energy_grad(x, y, t, BIG, margin)
        hw = 0.5 * (np.abs(np.cos(t)) + np.abs(np.sin(t)))
        dhw = 0.5 * (-np.sign(np.cos(t)) * np.sin(t) + np.sign(np.sin(t)) * np.cos(t))
        py = np.maximum(y + hw + margin - self.H, 0)
        e += (py ** 2).sum(); gy = gy + 2 * py; gt = gt + 2 * py * dhw
        hu = hext(t, self.psi)
        pu = np.maximum(x * self.cs - y * self.sn + hu + margin - c, 0)
        e += (pu ** 2).sum()
        gx = gx + 2 * pu * self.cs; gy = gy - 2 * pu * self.sn; gt = gt + 2 * pu * dhext(t, self.psi)
        gc = -2 * pu.sum()
        return e, gx, gy, gt, gc, worst

    def feasible(self, x, y, t, c, margin=MARGIN):
        e, *_, worst = self.energy(x, y, t, c, 0.0)
        hw = 0.5 * (np.abs(np.cos(t)) + np.abs(np.sin(t)))
        ok_w = np.all(x - hw >= -1e-12) and np.all(y - hw >= -1e-12) and np.all(y + hw <= self.H + 1e-12)
        ok_u = np.all(x * self.cs - y * self.sn + hext(t, self.psi) <= c + 1e-12)
        return worst <= 0 and ok_w and ok_u

    def waste(self, N, c):
        return self.m * (c + self.sn * self.H / 2) - N

    def compact(self, x, y, t, c, mus=(10.0, 100.0, 1e3, 1e4, 1e5), maxiter=500):
        n = len(x)
        z = np.concatenate([x, y, t, [c]])
        for mu in mus:
            def obj(z):
                e, gx, gy, gt, gc, _ = self.energy(z[:n], z[n:2 * n], z[2 * n:3 * n], z[-1])
                return z[-1] + mu * e, np.concatenate([mu * gx, mu * gy, mu * gt, [1 + mu * gc]])
            z = minimize(obj, z, jac=True, method='L-BFGS-B', options={'maxiter': maxiter, 'maxcor': 20}).x
        return z[:n], z[n:2 * n], z[2 * n:3 * n], float(z[-1])

    def relax(self, x, y, t, c, maxiter=2000):
        n = len(x)
        z = np.concatenate([x, y, t])
        def obj(z):
            e, gx, gy, gt, _, _ = self.energy(z[:n], z[n:2 * n], z[2 * n:], c)
            return e, np.concatenate([gx, gy, gt])
        z = minimize(obj, z, jac=True, method='L-BFGS-B', options={'maxiter': maxiter, 'maxcor': 20, 'ftol': 1e-30, 'gtol': 1e-14}).x
        return z[:n], z[n:2 * n], z[2 * n:]

    def squeeze(self, x, y, t, c, rounds=30, step=0.01):
        for _ in range(40):
            x, y, t = self.relax(x, y, t, c)
            if self.feasible(x, y, t, c):
                break
            c += 0.01
        else:
            return None
        best = (x, y, t, c)
        for _ in range(rounds):
            if step < 1e-8:
                break
            bx, by, bt, bc = best
            ct = bc - step
            # shift squares left proportionally to their u-coordinate to make room
            nx, ny, nt = self.relax(bx - step * 0.5, by, bt, ct)
            if self.feasible(nx, ny, nt, ct):
                best = (nx, ny, nt, ct); step *= 1.5
            else:
                step *= 0.4
        return best
