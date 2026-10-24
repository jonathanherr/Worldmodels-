"""Squares in Squares: a simulated world for packing n unit squares into a small square.

Problem (Kaggle "Squares in Squares"): place n unit squares, each with a center (x, y) and a
rotation angle, without overlapping, so that the smallest axis-aligned square enclosing all of
them has the shortest possible side L. A puzzle scores L / sqrt(n) - 1 (lower is better; the
sqrt(n) bound is only reachable for perfect squares).

This module has three layers:

1. Geometry and scoring (`corners`, `pair_overlaps`, `enclosing_side`, `validate`, `puzzle_score`)
   that work on plain arrays of x, y, degrees and can check a candidate solution.
2. A soft-body relaxation that pushes overlapping squares apart, keeps them inside a container
   wall, and lets contact forces turn them (`relax`).
3. `SquarePackingEnv`, a Gym-style world on top of the relaxation, in the style of `wm/env.py`:
   an agent nudges and rotates one selected square at a time and squeezes the container. It
   renders 64x64 frames and exposes ground-truth state, so it can feed a learned world model.
"""

from dataclasses import dataclass, field

import numpy as np

HALF = 0.5  # half the side of a unit square
_LOCAL = np.array([[-HALF, -HALF], [HALF, -HALF], [HALF, HALF], [-HALF, HALF]])

# Best known side lengths s(n) from Friedman's survey "Packing Unit Squares in Squares".
# n <= 10 are proven optimal; 11 is the best known packing.
KNOWN_BEST = {
    1: 1.0, 2: 2.0, 3: 2.0, 4: 2.0, 5: 2 + 1 / np.sqrt(2), 6: 3.0, 7: 3.0, 8: 3.0, 9: 3.0,
    10: 3 + 1 / np.sqrt(2), 11: 3.8771,
}


# ----------------------------------------------------------------------------- geometry


def rotations(deg: np.ndarray) -> np.ndarray:
    """(n, 2, 2) rotation matrices for angles in degrees (counter-clockwise)."""
    t = np.deg2rad(np.asarray(deg, dtype=np.float64))
    c, s = np.cos(t), np.sin(t)
    return np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)


def corners(x, y, deg) -> np.ndarray:
    """(n, 4, 2) corner coordinates of unit squares centered at (x, y) rotated by deg."""
    centers = np.stack([np.asarray(x, np.float64), np.asarray(y, np.float64)], -1)
    return centers[:, None, :] + np.einsum("nij,kj->nki", rotations(deg), _LOCAL)


def enclosing_side(x, y, deg) -> float:
    """Side of the smallest axis-aligned square that contains every unit square."""
    c = corners(x, y, deg).reshape(-1, 2)
    span = c.max(0) - c.min(0)
    return float(span.max())


def puzzle_score(side: float, n: int) -> float:
    return side / np.sqrt(n) - 1.0


def pair_overlaps(x, y, deg):
    """Separating-axis test for every pair of squares.

    Returns (i, j, depth, normal): for each pair i < j whose centers are close enough to touch,
    the penetration depth along the best separating axis (<= 0 means separated or touching)
    and the unit normal pointing from square i to square j.
    """
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    n = len(x)
    i, j = np.triu_indices(n, 1)
    centers = np.stack([x, y], -1)
    d = centers[j] - centers[i]
    near = (d**2).sum(-1) < 2.0 + 1e-9  # two unit squares can only touch within sqrt(2)
    i, j, d = i[near], j[near], d[near]
    if len(i) == 0:
        return i, j, np.zeros(0), np.zeros((0, 2))
    rot = rotations(deg)
    axes = np.concatenate([rot[i].transpose(0, 2, 1), rot[j].transpose(0, 2, 1)], 1)  # (P, 4, 2)
    # Each square projects onto an axis as an interval center +- radius.
    rad_i = HALF * np.abs(np.einsum("pak,pbk->pab", axes, rot[i].transpose(0, 2, 1))).sum(-1)
    rad_j = HALF * np.abs(np.einsum("pak,pbk->pab", axes, rot[j].transpose(0, 2, 1))).sum(-1)
    dist = np.einsum("pak,pk->pa", axes, d)
    overlap = rad_i + rad_j - np.abs(dist)  # (P, 4)
    best = overlap.argmin(1)
    rows = np.arange(len(i))
    depth = overlap[rows, best]
    normal = axes[rows, best] * np.sign(dist[rows, best] + 1e-12)[:, None]
    return i, j, depth, normal


def validate(x, y, deg, tol: float = 1e-9) -> dict:
    """Check a candidate solution: no two squares may overlap by more than tol.

    Returns a dict with the enclosing side, the score and the worst overlap found. This is our
    own checker, not Kaggle's official metric, so treat a pass here as necessary, not sufficient.
    """
    n = len(x)
    _, _, depth, _ = pair_overlaps(x, y, deg)
    worst = float(depth.max()) if len(depth) else 0.0
    side = enclosing_side(x, y, deg)
    return {
        "n": n,
        "valid": worst <= tol,
        "worst_overlap": worst,
        "side": side,
        "score": puzzle_score(side, n),
    }


def grid_solution(n: int):
    """The trivial baseline: a ceil(sqrt(n)) x ceil(sqrt(n)) grid, no rotation."""
    k = int(np.ceil(np.sqrt(n)))
    idx = np.arange(n)
    return idx % k + HALF, idx // k + HALF, np.zeros(n)


def five_solution():
    """The proven optimum for n = 5: four corner squares and one at 45 degrees in the middle."""
    s = 2 + 1 / np.sqrt(2)
    x = np.array([HALF, s - HALF, HALF, s - HALF, s / 2])
    y = np.array([HALF, HALF, s - HALF, s - HALF, s / 2])
    return x, y, np.array([0, 0, 0, 0, 45.0])


def polish(x, y, deg, gap: float = 1e-9, max_iter: int = 200):
    """Turn a nearly feasible packing into a strictly valid one by spreading centers apart.

    The soft physics leaves overlaps up to SquaresConfig.feasible_tol. Scaling every center away
    from the middle by a tiny factor removes them at a matching cost in side length. Kaggle's
    metric certifies non-overlap with interval arithmetic, so squares that merely touch in float64
    can be rejected; we keep every pair at least `gap` apart instead. Returns (x, y, deg, report)
    with the enclosing square moved to start at the origin.
    """
    tol = -gap
    x, y, deg = (np.array(a, dtype=np.float64) for a in (x, y, deg))
    cx, cy = x.mean(), y.mean()
    scale = 1.0
    for _ in range(max_iter):
        sx, sy = cx + (x - cx) * scale, cy + (y - cy) * scale
        report = validate(sx, sy, deg, tol)
        if report["valid"]:
            break
        scale *= 1 + max(report["worst_overlap"] + gap, 1e-12)
    lo = corners(sx, sy, deg).reshape(-1, 2).min(0)
    return sx - lo[0], sy - lo[1], deg, report


# Columns of the competition's sample_submission.csv: a running row id, the puzzle size n, the
# square's index within the puzzle, its center, and its angle in radians. A full submission has
# one block per n = 1..1000 (500,500 rows).
SUBMISSION_COLUMNS = ["id", "n", "square_id", "x", "y", "angle"]
MAX_N = 1000


def to_submission_rows(solutions: dict, fill_to: int | None = MAX_N) -> list[dict]:
    """Flatten {n: (x, y, deg)} into submission rows, converting degrees to radians.

    Puzzles missing from `solutions` up to `fill_to` get the grid packing, so the file is
    complete (the sample submission is exactly this grid). Pass fill_to=None to skip that.
    """
    sizes = sorted(set(solutions) | set(range(1, (fill_to or 0) + 1)))
    rows = []
    for n in sizes:
        x, y, deg = solutions[n] if n in solutions else grid_solution(n)
        for k in range(n):
            rows.append({"id": len(rows), "n": n, "square_id": k, "x": float(x[k]), "y": float(y[k]),
                         "angle": float(np.deg2rad(deg[k]))})
    return rows


def official_check(rows: list[dict]) -> dict:
    """Score one puzzle's submission rows with Kaggle's own metric (wm/kaggle_metric.py).

    The rows go through the same decimal-string round trip as a CSV file, so this is exactly
    what the leaderboard would compute for that puzzle, interval-arithmetic overlap check included.
    """
    from . import kaggle_metric

    n = rows[0]["n"]
    as_text = [{k: repr(v) if isinstance(v, float) else str(v) for k, v in r.items()} for r in rows]
    _, score, side, reason = kaggle_metric.evaluate_single_puzzle(n, as_text)
    return {"valid": reason == "ok", "score": score, "side": float(side) if side else None, "reason": reason}


# ----------------------------------------------------------------------------- physics


@dataclass
class PhysicsConfig:
    iters: int = 8  # relaxation passes per env step
    push: float = 0.5  # fraction of each overlap resolved per pass
    turn: float = 0.15  # how strongly off-center contacts rotate a square
    max_turn_deg: float = 4.0  # cap on rotation per pass, keeps things stable


def _inside(points: np.ndarray, x, y, deg, idx: np.ndarray) -> np.ndarray:
    """Mask of which points (P, 4, 2) lie inside square idx[p]."""
    rot = rotations(np.asarray(deg)[idx])
    local = np.einsum("pji,pkj->pki", rot, points - np.stack([x[idx], y[idx]], -1)[:, None, :])
    return (np.abs(local) <= HALF + 1e-9).all(-1)


def relax(x, y, deg, side: float, cfg: PhysicsConfig | None = None, pinned: int | None = None):
    """Resolve overlaps and wall penetration by small pushes, with contact torque.

    Squares push apart along the separating-axis normal. The push acts at the mean of the
    penetrating corners, so an off-center contact also turns the squares, which is what lets
    tilted packings appear on their own. The container is [0, side]^2. `pinned` (the square the
    agent is moving) receives half the push of the others so that it is not just shoved back.
    """
    cfg = cfg or PhysicsConfig()
    x, y, deg = (np.array(a, dtype=np.float64) for a in (x, y, deg))
    n = len(x)
    weight = np.ones(n)
    if pinned is not None:
        weight[pinned] = 0.5
    for _ in range(cfg.iters):
        dx, dy, dturn = np.zeros(n), np.zeros(n), np.zeros(n)
        # Square-square contacts.
        i, j, depth, normal = pair_overlaps(x, y, deg)
        hit = depth > 0
        i, j, depth, normal = i[hit], j[hit], depth[hit], normal[hit]
        if len(i):
            c = corners(x, y, deg)
            in_j = _inside(c[i], x, y, deg, j)  # corners of i inside j
            in_i = _inside(c[j], x, y, deg, i)
            pts = np.concatenate([c[i], c[j]], 1)
            mask = np.concatenate([in_j, in_i], 1).astype(np.float64)
            cnt = mask.sum(1, keepdims=True)
            mid = 0.5 * (np.stack([x[i], y[i]], -1) + np.stack([x[j], y[j]], -1))
            contact = np.where(cnt > 0, (pts * mask[..., None]).sum(1) / np.maximum(cnt, 1), mid)
            f = normal * (cfg.push * depth)[:, None]
            share_i = weight[i] / (weight[i] + weight[j])
            share_j = 1 - share_i
            for idx, sign, share in ((i, -1.0, share_i), (j, 1.0, share_j)):
                force = sign * f * share[:, None]
                np.add.at(dx, idx, force[:, 0])
                np.add.at(dy, idx, force[:, 1])
                r = contact - np.stack([x[idx], y[idx]], -1)
                np.add.at(dturn, idx, r[:, 0] * force[:, 1] - r[:, 1] * force[:, 0])
        # Walls.
        c = corners(x, y, deg)
        pen = np.clip(-c, 0, None) - np.clip(c - side, 0, None)  # (n, 4, 2), inward correction
        wall_shift = pen.max(1) + pen.min(1)  # largest correction per axis and direction
        r = c - np.stack([x, y], -1)[:, None, :]
        dturn += cfg.push * (r[..., 0] * pen[..., 1] - r[..., 1] * pen[..., 0]).sum(1)
        dx += wall_shift[:, 0] * weight
        dy += wall_shift[:, 1] * weight
        x, y = x + dx, y + dy
        # Torque over the unit square's moment of inertia (1/6), capped.
        deg = deg + np.clip(np.rad2deg(6 * cfg.turn * dturn), -cfg.max_turn_deg, cfg.max_turn_deg)
    deg = (deg + 45.0) % 90.0 - 45.0  # a square looks the same every 90 degrees
    return x, y, deg


# ----------------------------------------------------------------------------- environment

NOOP, NEXT, UP, DOWN, LEFT, RIGHT, CCW, CW, SQUEEZE, SHAKE = range(10)
NUM_ACTIONS = 10
ACTION_NAMES = ["noop", "next", "up", "down", "left", "right", "ccw", "cw", "squeeze", "shake"]

SQUARE_VALUE = 0.5
SELECTED_VALUE = 0.75
OVERLAP_VALUE = 1.0
WALL_VALUE = 0.25


@dataclass
class SquaresConfig:
    n: int = 5
    size: int = 64  # rendered frame is size x size
    start_slack: float = 1.4  # starting container side = start_slack * ceil(sqrt(n))
    move: float = 0.08  # nudge per move action, in square sides
    turn_deg: float = 5.0
    squeeze: float = 0.02  # container shrink per squeeze action
    shake: float = 0.05  # std of the random kick from shake
    feasible_tol: float = 1e-3  # overlap the soft physics tolerates when calling a state feasible
    max_steps: int = 300
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)


class SquarePackingEnv:
    """Gym-style API: reset() -> obs, step(a) -> obs, reward, done, info.

    The container is the square [0, side]^2. Squeezing shrinks it by `squeeze`, and the
    relaxation then pushes the squares back inside and apart. A state is feasible when the
    remaining overlap is below `feasible_tol`. Reward is the drop in the best feasible puzzle
    score seen so far in the episode, so the return of an episode is start score - best score.

    Frames are single-channel float32 in [0, 1] with a fixed world-to-pixel scale set by the
    starting container: walls at 0.25, squares at 0.5, the selected square at 0.75 and any
    overlapping area at 1.0, so overlaps are visible from pixels alone.
    """

    def __init__(self, config: SquaresConfig | None = None, seed: int | None = None):
        self.cfg = config or SquaresConfig()
        self.rng = np.random.default_rng(seed)
        s = self.cfg.size * 2  # render at 2x and average down for anti-aliasing
        ys, xs = np.mgrid[0:s, 0:s]
        self._px = (xs.astype(np.float64) + 0.5) / s
        self._py = 1.0 - (ys.astype(np.float64) + 0.5) / s  # y up, like the maths

    # Ground-truth state: n rows of [x, y, deg] then a final row [side, selected, 0].
    def state(self) -> np.ndarray:
        squares = np.stack([self.x, self.y, self.deg], -1)
        return np.concatenate([squares, [[self.side, self.selected, 0.0]]]).astype(np.float32)

    def reset(self) -> np.ndarray:
        c = self.cfg
        n = c.n
        self.t = 0
        self.side = c.start_slack * np.ceil(np.sqrt(n))
        self.world = self.side * 1.05  # fixed pixel scale for the whole episode
        self.x = self.rng.uniform(HALF, self.side - HALF, n)
        self.y = self.rng.uniform(HALF, self.side - HALF, n)
        self.deg = self.rng.uniform(-45, 45, n)
        self.selected = 0
        for _ in range(10):
            self._relax()
        self.best = self._score() if self.feasible() else puzzle_score(self.side, n)
        self.best_solution = self.solution()
        return self.render()

    def _relax(self):
        self.x, self.y, self.deg = relax(self.x, self.y, self.deg, self.side, self.cfg.physics, self.selected)

    def overlap(self) -> float:
        """Worst square-square or square-wall penetration."""
        _, _, depth, _ = pair_overlaps(self.x, self.y, self.deg)
        c = corners(self.x, self.y, self.deg)
        wall = max(float(-c.min()), float(c.max() - self.side), 0.0)
        return max(float(depth.max()) if len(depth) else 0.0, wall)

    def feasible(self) -> bool:
        return self.overlap() <= self.cfg.feasible_tol

    def _score(self) -> float:
        return puzzle_score(enclosing_side(self.x, self.y, self.deg), self.cfg.n)

    def step(self, action: int):
        c = self.cfg
        self.t += 1
        k = self.selected
        if action == NEXT:
            self.selected = (k + 1) % c.n
        elif action in (UP, DOWN, LEFT, RIGHT):
            dx, dy = {UP: (0, 1), DOWN: (0, -1), LEFT: (-1, 0), RIGHT: (1, 0)}[action]
            self.x[k] += dx * c.move
            self.y[k] += dy * c.move
        elif action in (CCW, CW):
            self.deg[k] += c.turn_deg if action == CCW else -c.turn_deg
        elif action == SQUEEZE:
            self.side = max(self.side - c.squeeze, 1.0)
        elif action == SHAKE:
            self.x += self.rng.normal(0, c.shake, c.n)
            self.y += self.rng.normal(0, c.shake, c.n)
            self.deg += self.rng.normal(0, c.shake * 30, c.n)
        self._relax()

        reward = 0.0
        feasible = self.feasible()
        if feasible:
            score = self._score()
            if score < self.best:
                reward = self.best - score
                self.best = score
                self.best_solution = self.solution()
        done = self.t >= c.max_steps
        info = {"state": self.state(), "feasible": feasible, "overlap": self.overlap(), "best_score": self.best}
        return self.render(), reward, done, info

    def solution(self):
        """Current squares as (x, y, deg), shifted so the enclosing square starts at 0."""
        c = corners(self.x, self.y, self.deg).reshape(-1, 2)
        lo = c.min(0)
        return self.x - lo[0], self.y - lo[1], self.deg.copy()

    def render(self) -> np.ndarray:
        cfg, n = self.cfg, self.cfg.n
        wx, wy = self._px * self.world, self._py * self.world
        coverage = np.zeros(wx.shape, dtype=np.int32)
        frame = np.zeros(wx.shape, dtype=np.float64)
        rot = rotations(self.deg)
        for k in range(n):
            # Pixel centers in the square's own frame.
            u = rot[k, 0, 0] * (wx - self.x[k]) + rot[k, 1, 0] * (wy - self.y[k])
            v = rot[k, 0, 1] * (wx - self.x[k]) + rot[k, 1, 1] * (wy - self.y[k])
            inside = (np.abs(u) <= HALF) & (np.abs(v) <= HALF)
            coverage += inside
            frame[inside] = np.maximum(frame[inside], SELECTED_VALUE if k == self.selected else SQUARE_VALUE)
        frame[coverage > 1] = OVERLAP_VALUE
        # Container walls, one world-pixel thick, drawn under the squares.
        px = self.world / wx.shape[0]
        on_wall = ((np.abs(wx - self.side) < px) & (wy <= self.side)) | ((np.abs(wy - self.side) < px) & (wx <= self.side))
        on_wall |= ((wx < px) & (wy <= self.side)) | ((wy < px) & (wx <= self.side))
        frame = np.where((frame == 0) & on_wall, WALL_VALUE, frame)
        s = cfg.size
        return frame.reshape(s, 2, s, 2).mean((1, 3)).astype(np.float32)


# ----------------------------------------------------------------------------- policies


def squeeze_policy(env: SquarePackingEnv, rng: np.random.Generator) -> int:
    """A simple annealing heuristic: squeeze while feasible, otherwise wait, sometimes shake."""
    if env.feasible():
        return SQUEEZE
    return SHAKE if rng.random() < 0.15 else NOOP


def sticky_random_policy(rng: np.random.Generator, prev: int, repeat_prob: float = 0.7) -> int:
    if rng.random() < repeat_prob:
        return prev
    return int(rng.integers(NUM_ACTIONS))


def collect(num_steps: int, n: int = 5, seed: int = 0, squeeze_frac: float = 0.5) -> dict:
    """Collect transitions in the same layout as wm/data.collect.

    Each episode either follows the squeeze heuristic mixed with random actions (so the data
    covers tight, contact-rich packings) or a sticky random policy. The `state` field is the
    flattened ground-truth state from SquarePackingEnv.state().
    """
    env = SquarePackingEnv(SquaresConfig(n=n), seed=seed)
    rng = np.random.default_rng(seed + 1)
    obs, act, rew, done, state, starts = [], [], [], [], [], []
    while len(obs) < num_steps:
        starts.append(len(obs))
        o, a, d = env.reset(), int(rng.integers(NUM_ACTIONS)), False
        squeezy = rng.random() < squeeze_frac
        while not d:
            if squeezy and rng.random() < 0.7:
                a = squeeze_policy(env, rng)
            else:
                a = sticky_random_policy(rng, a)
            s = env.state()
            o2, r, d, _ = env.step(a)
            obs.append(o); act.append(a); rew.append(r); done.append(d); state.append(s.ravel())
            o = o2
        obs.append(o); act.append(NOOP); rew.append(0.0); done.append(True); state.append(env.state().ravel())
    return {
        "obs": (np.stack(obs) * 255).astype(np.uint8),
        "action": np.array(act, dtype=np.int64),
        "reward": np.array(rew, dtype=np.float32),
        "done": np.array(done, dtype=bool),
        "state": np.stack(state).astype(np.float32),
        "episode_starts": np.array(starts + [len(obs)], dtype=np.int64),
    }
