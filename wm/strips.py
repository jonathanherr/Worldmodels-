"""Tilted strips at any angle (Erdos-Graham / Goebel style) for n just below a perfect square.

A band of width W and height H = r + delta (delta < 1) holds only r * floor(W) axis-aligned
squares. Tilting columns of r + 1 squares by a small angle theta makes each column
(r + 1) cos(theta) + sin(theta) tall and 1 / cos(theta) wide, so per unit width the band holds
(r + 1) cos(theta) squares instead of r. For wide bands and delta above roughly
sqrt(2 / (r + 1)) that beats the grid, minus the wedges lost where the columns meet the walls.

The square container [0, S]^2 is cut into an axis-aligned A x B grid block in the bottom-left
corner, a top band [0, S] x [B, S] and a right band [A, S] x [0, B]; each band takes the best
tilted-column fill (angle theta, column phase t0). The 45-degree constructions in
wm/constructions.py only use theta = 45 degrees; this family uses every angle.

Columns: rotate the plane by theta (p along the column normal, q along the column). Column j
occupies p in [t0 + j, t0 + j + 1]. The band is convex, so its lower boundary lo(p) is convex and
its upper boundary hi(p) concave, and the longest q-interval that fits for p in [t, t + 1] is
min(hi(t), hi(t + 1)) - max(lo(t), lo(t + 1)).
"""

import numpy as np

GAP = 2e-9  # spacing between neighbouring squares and from walls, so nothing touches exactly


def _chains(W, H, c, s):
    """Lower and upper boundary of the W x H rectangle in rotated (p, q) coordinates."""
    lp = np.array([0.0, W * c, W * c + H * s])
    lq = np.array([0.0, -W * s, -W * s + H * c])
    up = np.array([0.0, H * s, W * c + H * s])
    uq = np.array([0.0, H * c, -W * s + H * c])
    return lp, lq, up, uq


def band_columns(W, H, theta, t0):
    """Per column: (t, q_lo, count) for the tilted-column fill of a W x H band (0 < theta < 90 deg)."""
    c, s = np.cos(theta), np.sin(theta)
    lp, lq, up, uq = _chains(W, H, c, s)
    pmax = W * c + H * s
    pitch = 1 + GAP
    t = t0 + GAP + pitch * np.arange(int(pmax / pitch) + 2)
    t = t[t + 1 <= pmax - GAP]
    lo = np.maximum(np.interp(t, lp, lq), np.interp(t + 1, lp, lq)) + GAP
    hi = np.minimum(np.interp(t, up, uq), np.interp(t + 1, up, uq)) - GAP
    cnt = np.floor((hi - lo + GAP) / pitch + 1e-12).astype(int)
    cnt = np.maximum(cnt, 0)
    return t, lo, cnt


def band_count(W, H, theta, t0):
    if W < 1 or H < 1:
        return 0
    return int(band_columns(W, H, theta, t0)[2].sum())


def best_band(W, H, thetas=None, phases=16):
    """(count, theta, t0) of the best fill of a W x H band; theta = 0 means plain grid."""
    if W < 1 - 1e-12 or H < 1 - 1e-12:
        return 0, 0.0, 0.0
    grid = int(np.floor(W + 1e-12)) * int(np.floor(H + 1e-12))
    if thetas is None:
        thetas = np.deg2rad(np.linspace(0.1, 45, 900))
    th = np.asarray(thetas)[:, None, None]
    t0 = np.linspace(0, 1, phases, endpoint=False)[None, :, None]
    c, s = np.cos(th), np.sin(th)
    pmax = W * c + H * s
    pitch = 1 + GAP
    t = t0 + GAP + pitch * np.arange(int((W + H) / pitch) + 2)[None, None, :]
    # Boundary chains, evaluated piecewise: lower = (0,0)-(Wc,-Ws)-(pmax,-Ws+Hc), upper via (Hs,Hc).
    def lower(p):
        return np.where(p <= W * c, -p * s / c, -W * s + (p - W * c) * c / s)
    def upper(p):
        return np.where(p <= H * s, p * c / s, H * c - (p - H * s) * s / c)
    lo = np.maximum(lower(t), lower(t + 1)) + GAP
    hi = np.minimum(upper(t), upper(t + 1)) - GAP
    cnt = np.floor((hi - lo + GAP) / pitch + 1e-12)
    cnt = np.where((t + 1 <= pmax - GAP) & (cnt > 0), cnt, 0).sum(-1)
    i, j = np.unravel_index(np.argmax(cnt), cnt.shape)
    if cnt[i, j] <= grid:
        return grid, 0.0, 0.0
    return int(cnt[i, j]), float(th[i, 0, 0]), float(t0[0, j, 0])


def band_squares(W, H, theta, t0, x0=0.0, y0=0.0):
    """Centers and angles (degrees) of the tilted fill of the band placed at (x0, y0)."""
    if theta == 0.0:
        gx = int(np.floor(W + 1e-12))
        gy = int(np.floor(H + 1e-12))
        i, j = np.meshgrid(np.arange(gx), np.arange(gy), indexing="ij")
        x = x0 + GAP + i.ravel() * (1 + GAP) + 0.5
        y = y0 + GAP + j.ravel() * (1 + GAP) + 0.5
        return x, y, np.zeros(len(x))
    c, s = np.cos(theta), np.sin(theta)
    t, lo, cnt = band_columns(W, H, theta, t0)
    P, Q = [], []
    for tj, qj, kj in zip(t, lo, cnt):
        P += [tj + 0.5] * kj
        Q += list(qj + 0.5 + (1 + GAP) * np.arange(kj))
    P, Q = np.array(P), np.array(Q)
    return x0 + P * c - Q * s, y0 + P * s + Q * c, np.full(len(P), np.rad2deg(theta))


def edge(k):
    """Right/top edge of a k-wide grid block laid out by band_squares, plus one gap."""
    return k * (1 + GAP) + 2 * GAP


def layout(S, A, B, top, right):
    """Packing for container side S: A x B grid, top band fill `top` = (theta, t0), right band `right`.
    The right band is the top-band construction transposed (x <-> y), so its columns lie horizontally."""
    gx, gy, gd = band_squares(A, B, 0.0, 0.0)
    tx, ty, td = band_squares(S, S - edge(B), *top, 0.0, edge(B))
    # Right band W = S - A, H = B: build as a B x (S - A) band and transpose.
    ry, rx, rd = band_squares(edge(B) - GAP, S - edge(A), *right, 0.0, edge(A))
    rd = -rd if right[0] else rd  # transposing mirrors the rotation
    return (np.concatenate([gx, tx, rx]), np.concatenate([gy, ty, ry]), np.concatenate([gd, td, rd]))


def layout_count(S, A, B, thetas=None, phases=16):
    """(n, top_fill, right_fill) for the best band fills with container side S and an A x B grid block."""
    top = best_band(S, S - edge(B), thetas, phases)
    right = best_band(edge(B) - GAP, S - edge(A), thetas, phases)
    return A * B + top[0] + right[0], top[1:], right[1:]
