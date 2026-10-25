"""Which number-theoretic properties of n predict how well n packs? Reads the best-per-n store.

    SQUARES_BEST_DIR=/mnt/project-files/squares/best python -m scripts.number_theory_squares

Writes n = k^2 + d (k = floor sqrt n, 0 <= d <= 2k) and u = d / (2k + 1), the position between
consecutive squares. Prints
* score and the fraction of n that beat the grid, by u bucket (position);
* for labels like prime, even, 4 | n: the mean score residual against same-row neighbours
  (n +- 2 with the same k), with a permutation p-value, so position is controlled for;
* fixed-offset families: s((k+1)^2 + d) <= s(k^2 + d) + 1 (wrap an L of 2k + 1 squares) and
  how often the larger k is strictly better;
* tilings: s(j^2 m) <= j s(m) and the density m / s(m) of every small packing against grid.
"""

import math

import numpy as np

from wm import best_store

N = 1000


def isprime(m):
    return m > 1 and all(m % p for p in range(2, math.isqrt(m) + 1))


def sides():
    s = np.array([0.0] + [best_store.best_side(n) for n in range(1, N + 1)])
    for n in range(N - 1, 0, -1):  # a packing of n + 1 squares also packs n
        s[n] = min(s[n], s[n + 1])
    return s


def main():
    s = sides()
    n = np.arange(N + 1)
    sc = s / np.sqrt(np.maximum(n, 1)) - 1
    k = np.floor(np.sqrt(n)).astype(int)
    d = n - k * k
    grid = np.ceil(np.sqrt(np.maximum(n, 1)))
    print(f"store total {sc[1:].sum():.6f}, n beating grid {int((s[1:] < grid[1:] - 1e-9).sum())}")

    print("\nposition u = (n - k^2) / (2k + 1):")
    u = d / (2 * k + 1)
    for lo in np.arange(0, 1, 0.1):
        m = (u >= lo) & (u < lo + 0.1) & (n >= 10)
        print(f"  u {lo:.1f}-{lo + .1:.1f}: mean score {sc[m].mean():.4f}, beats grid {np.mean(s[m] < grid[m] - 1e-9):.2f}")

    ok = (n >= 10) & (d > 0)
    res = np.full(N + 1, np.nan)
    for i in np.where(ok)[0]:
        nb = [j for j in range(i - 2, i + 3) if j != i and j <= N and k[j] == k[i] and d[j] > 0]
        res[i] = sc[i] - np.mean(sc[nb])
    idx = np.where(ok)[0]
    rng = np.random.default_rng(0)
    print("\nlabel effects after removing position (residual vs same-row neighbours, permutation p):")
    labels = {
        "prime": isprime,
        "even": lambda i: i % 2 == 0,
        "4 | n": lambda i: i % 4 == 0,
        "3 | n": lambda i: i % 3 == 0,
        "squarefree": lambda i: all(i % (p * p) for p in range(2, math.isqrt(i) + 1)),
        ">= 12 divisors": lambda i: sum(i % q == 0 for q in range(1, i + 1)) >= 12,
    }
    r = res[idx]
    for name, f in labels.items():
        lab = np.array([f(i) for i in idx])
        obs = r[lab].mean() - r[~lab].mean()
        perm = []
        for _ in range(2000):
            p = rng.permutation(lab)
            perm.append(r[p].mean() - r[~p].mean())
        pv = np.mean(np.abs(perm) >= abs(obs))
        print(f"  {name:15s} count {lab.sum():4d}  residual diff {obs:+.5f}  p={pv:.3f}")

    print("\nfixed-offset families k^2 + d -> (k+1)^2 + d:")
    viol, better, beaten = [], 0, 0
    for kk in range(2, 32):
        for dd in range(0, 2 * kk + 1):
            a, b = kk * kk + dd, (kk + 1) ** 2 + dd
            if b > N:
                continue
            e1, e2 = s[a] - kk, s[b] - kk - 1
            beaten += e1 < 1 - 1e-9
            better += e2 < e1 - 1e-6
            if e2 > e1 + 1e-7:
                viol.append((b, a))
    print(f"  L-wrap violations (should be 0 after compose_squares): {viol}")
    print(f"  larger k strictly better in {better} of {beaten} beaten steps")

    print("\ntilings s(j^2 m) <= j s(m):")
    wins = [(j * j * m, m, j) for m in range(2, N + 1) for j in range(2, 32)
            if j * j * m <= N and j * s[m] < s[j * j * m] - 1e-7]
    print(f"  wins: {wins or 'none'}")
    worst = max(m / s[m] - math.floor(s[m]) for m in range(2, 201) if s[m] % 1 > 1e-9)
    print(f"  best m <= 200 packing density minus grid density per unit length: {worst:+.3f} "
          "(negative: repeating a square packing along a strip never beats grid columns)")


if __name__ == "__main__":
    main()
