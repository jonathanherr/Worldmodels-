"""A small top-down video game world: four themed rooms, coins, patrolling slimes.

The world is a fixed 24x24 tile map (8 px tiles). The agent only sees a 64x64 RGB view
(8x8 tiles) centred on the player, so the world model has to learn the map itself: what
lies beyond the edge of the screen, which coins it already took, and where slimes walk.

Actions: 0 stay, 1 up, 2 down, 3 left, 4 right. One tile per step.
Reward: +1 per coin. Touching a slime gives -1 and ends the episode.
"""

from __future__ import annotations

import numpy as np

STAY, UP, DOWN, LEFT, RIGHT = range(5)
NUM_ACTIONS = 5
MOVES = {STAY: (0, 0), UP: (0, -1), DOWN: (0, 1), LEFT: (-1, 0), RIGHT: (1, 0)}

TILE = 8
MAP = 24
VIEW = 8  # tiles per side
PAD = VIEW // 2

FLOOR, WALL, WATER, TREE, ROCK = range(5)
BLOCKING = {WALL, WATER, TREE, ROCK}

# Floor colour per room (quadrant), so the model can tell where it is.
ROOM_COLOURS = {
    0: (0.36, 0.62, 0.30),  # top-left: meadow
    1: (0.80, 0.70, 0.45),  # top-right: desert
    2: (0.55, 0.55, 0.60),  # bottom-left: stone hall
    3: (0.55, 0.38, 0.24),  # bottom-right: wooden cabin
}

# Each slime walks back and forth along a straight segment: (x0, y0, x1, y1).
SLIME_PATHS = [(14, 4, 20, 4), (4, 15, 4, 20)]


def room_of(x: int, y: int) -> int:
    return (1 if x >= MAP // 2 else 0) + (2 if y >= MAP // 2 else 0)


def build_map() -> np.ndarray:
    """Hand-made layout: outer wall, a cross of inner walls with doorways, room decorations."""
    m = np.zeros((MAP, MAP), dtype=np.int64)
    m[0, :] = m[-1, :] = m[:, 0] = m[:, -1] = WALL
    m[MAP // 2, :] = WALL
    m[:, MAP // 2] = WALL
    for x, y in [(6, 12), (17, 12), (12, 6), (12, 18), (5, 12), (18, 12)]:  # doorways
        m[y, x] = FLOOR
    # Meadow: trees.
    for x, y in [(3, 3), (4, 3), (8, 2), (9, 7), (3, 8), (7, 9), (2, 6)]:
        m[y, x] = TREE
    # Desert: a pond and rocks.
    m[7:10, 15:19] = WATER
    for x, y in [(20, 8), (14, 2), (21, 2)]:
        m[y, x] = ROCK
    # Stone hall: pillars.
    for x in (7, 9):
        for y in (15, 18, 21):
            m[y, x] = ROCK
    # Cabin: crates along the walls.
    for x, y in [(14, 14), (15, 14), (21, 21), (21, 20), (16, 19), (19, 16)]:
        m[y, x] = ROCK
    return m


COIN_SPOTS = [
    (2, 2), (6, 4), (10, 10), (5, 7), (8, 5),
    (15, 3), (20, 6), (14, 10), (19, 10), (22, 1),
    (2, 21), (6, 16), (10, 14), (8, 20),
    (15, 16), (20, 14), (17, 21), (22, 18),
]


def _tile_texture(kind: int, room: int, x: int, y: int, rng: np.random.Generator) -> np.ndarray:
    t = np.zeros((TILE, TILE, 3), dtype=np.float32)
    base = np.array(ROOM_COLOURS[room], dtype=np.float32)
    t[:] = base * (0.94 if (x + y) % 2 else 1.0)  # subtle checkerboard
    t += rng.normal(0, 0.02, (TILE, TILE, 1)).astype(np.float32)
    if kind == WALL:
        t[:] = (0.25, 0.22, 0.28)
        t[3, :] = t[7, :] = (0.15, 0.13, 0.17)  # brick mortar lines
        t[0:3, 4] = t[4:7, 0] = (0.15, 0.13, 0.17)
    elif kind == WATER:
        t[:] = (0.15, 0.35, 0.75)
        t[2, 1:4] = t[5, 4:7] = (0.45, 0.65, 0.95)  # ripples
    elif kind == TREE:
        t[5:8, 3:5] = (0.35, 0.22, 0.10)
        yy, xx = np.mgrid[0:TILE, 0:TILE]
        t[((xx - 3.5) ** 2 + (yy - 3) ** 2) < 9] = (0.10, 0.38, 0.12)
    elif kind == ROCK:
        yy, xx = np.mgrid[0:TILE, 0:TILE]
        t[((xx - 3.5) ** 2 / 12 + (yy - 4) ** 2 / 8) < 1] = (0.42, 0.40, 0.38)
    return np.clip(t, 0, 1)


def _disc(colour, radius, cx=3.5, cy=3.5):
    yy, xx = np.mgrid[0:TILE, 0:TILE] + 0.5
    mask = ((xx - 0.5 - cx) ** 2 + (yy - 0.5 - cy) ** 2) <= radius ** 2
    sprite = np.zeros((TILE, TILE, 3), dtype=np.float32)
    sprite[mask] = colour
    return sprite, mask


class GameWorld:
    """Gym-style API: reset() -> obs, step(a) -> obs, reward, done, info.

    obs is float32 (64, 64, 3) in [0, 1]. info["sprite_mask"] marks pixels covered by
    coins, slimes or the player (used only to weight the training loss, never as input).
    """

    def __init__(self, seed: int | None = None, max_steps: int = 256):
        self.rng = np.random.default_rng(seed)
        self.max_steps = max_steps
        self.map = build_map()
        tex_rng = np.random.default_rng(7)  # fixed so the world always looks the same
        world = np.zeros(((MAP + 2 * PAD) * TILE, (MAP + 2 * PAD) * TILE, 3), dtype=np.float32)
        for y in range(MAP):
            for x in range(MAP):
                py, px = (y + PAD) * TILE, (x + PAD) * TILE
                world[py : py + TILE, px : px + TILE] = _tile_texture(self.map[y, x], room_of(x, y), x, y, tex_rng)
        self.world_img = world  # outside the map stays black
        self.coin_sprite, self.coin_mask = _disc((1.0, 0.85, 0.1), 2.2)
        self.coin_sprite[self.coin_mask & (np.mgrid[0:TILE, 0:TILE][0] < 3)] = (1.0, 0.97, 0.6)  # shine
        self.slime_sprite, self.slime_mask = _disc((0.85, 0.12, 0.15), 3.3, cy=4.0)
        self.slime_sprite[3, 2] = self.slime_sprite[3, 5] = (1, 1, 1)
        self.floor_tiles = [(x, y) for y in range(MAP) for x in range(MAP)
                            if self.map[y, x] == FLOOR and (x, y) not in COIN_SPOTS]

    # ---------------------------------------------------------------- dynamics
    def _slime_positions(self, t: int):
        out = []
        for x0, y0, x1, y1 in SLIME_PATHS:
            n = max(abs(x1 - x0), abs(y1 - y0))
            k = t % (2 * n)
            k = k if k <= n else 2 * n - k
            out.append((x0 + np.sign(x1 - x0) * k, y0 + np.sign(y1 - y0) * k))
        return out

    def state(self) -> np.ndarray:
        """Ground truth: player x, y, facing, then one flag per coin (1 = still there)."""
        return np.array([self.x, self.y, self.facing, *self.coins], dtype=np.float32)

    def reset(self):
        self.t = int(self.rng.integers(0, 1000))  # random slime phase
        self.coins = [1] * len(COIN_SPOTS)
        slimes = self._slime_positions(self.t)
        while True:
            self.x, self.y = self.floor_tiles[self.rng.integers(len(self.floor_tiles))]
            if all(abs(self.x - sx) + abs(self.y - sy) > 3 for sx, sy in slimes):
                break
        self.facing = DOWN
        self.steps = 0
        return self.render()

    def step(self, action: int):
        self.steps += 1
        self.t += 1
        dx, dy = MOVES[action]
        if action != STAY:
            self.facing = action
            nx, ny = self.x + dx, self.y + dy
            if self.map[ny, nx] not in BLOCKING:
                self.x, self.y = nx, ny
        reward, done = 0.0, False
        for i, (cx, cy) in enumerate(COIN_SPOTS):
            if self.coins[i] and (cx, cy) == (self.x, self.y):
                self.coins[i] = 0
                reward += 1.0
        if (self.x, self.y) in self._slime_positions(self.t):
            reward, done = -1.0, True
        if self.steps >= self.max_steps:
            done = True
        obs, mask = self.render(with_mask=True)
        return obs, reward, done, {"state": self.state(), "sprite_mask": mask}

    # ---------------------------------------------------------------- rendering
    def render(self, with_mask: bool = False):
        x0, y0 = (self.x + PAD - VIEW // 2) * TILE, (self.y + PAD - VIEW // 2) * TILE
        frame = self.world_img[y0 : y0 + VIEW * TILE, x0 : x0 + VIEW * TILE].copy()
        mask = np.zeros(frame.shape[:2], dtype=bool)

        def blit(sprite, smask, tx, ty):
            vx, vy = tx - self.x + VIEW // 2, ty - self.y + VIEW // 2
            if 0 <= vx < VIEW and 0 <= vy < VIEW:
                region = frame[vy * TILE : (vy + 1) * TILE, vx * TILE : (vx + 1) * TILE]
                region[smask] = sprite[smask]
                mask[vy * TILE : (vy + 1) * TILE, vx * TILE : (vx + 1) * TILE] |= smask

        for i, (cx, cy) in enumerate(COIN_SPOTS):
            if self.coins[i]:
                blit(self.coin_sprite, self.coin_mask, cx, cy)
        for sx, sy in self._slime_positions(self.t):
            blit(self.slime_sprite, self.slime_mask, sx, sy)
        blit(*self._player_sprite(), self.x, self.y)
        return (frame, mask) if with_mask else frame

    def _player_sprite(self):
        sprite, mask = _disc((0.20, 0.45, 0.95), 3.2)
        sprite[1:3, 2:6] = (0.95, 0.80, 0.65)  # face
        mask[1:3, 2:6] = True
        ex, ey = {UP: (0, -1), DOWN: (0, 1), LEFT: (-1, 0), RIGHT: (1, 0)}[self.facing]
        for px in (2, 5):  # eyes look where the player last moved
            sprite[np.clip(4 + ey, 0, 7), np.clip(px + ex, 0, 7)] = (0.05, 0.05, 0.1)
        return sprite, mask


def explore_policy(env: GameWorld, rng: np.random.Generator, prev: int) -> int:
    """Wander: keep heading the same way, turn when blocked or at random."""
    if prev != STAY and rng.random() < 0.85:
        dx, dy = MOVES[prev]
        if env.map[env.y + dy, env.x + dx] not in BLOCKING:
            return prev
    if rng.random() < 0.05:
        return STAY
    return int(rng.integers(1, NUM_ACTIONS))
