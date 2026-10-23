"""A tiny 2D "breakout without bricks" environment with ground-truth state.

A ball bounces around a 64x64 box. The agent moves a paddle along the bottom.
Hitting the ball with the paddle gives +1, missing it gives -1 and ends the episode.

Frames are single-channel float32 in [0, 1]. The ball is drawn at intensity 1.0 and
the paddle at 0.5, so the two can be told apart from pixels alone (used in evaluation).
"""

from dataclasses import dataclass

import numpy as np

LEFT, STAY, RIGHT = 0, 1, 2
NUM_ACTIONS = 3

BALL_VALUE = 1.0
PADDLE_VALUE = 0.5


@dataclass
class EnvConfig:
    size: int = 64
    ball_radius: float = 2.5
    ball_speed: tuple = (1.5, 3.0)  # min, max speed in pixels per step
    paddle_width: int = 14
    paddle_height: int = 3
    paddle_speed: float = 3.0
    max_steps: int = 200


class BouncingBallEnv:
    """Gym-style API: reset() -> obs, step(a) -> obs, reward, done, info."""

    def __init__(self, config: EnvConfig | None = None, seed: int | None = None):
        self.cfg = config or EnvConfig()
        self.rng = np.random.default_rng(seed)
        s = self.cfg.size
        ys, xs = np.mgrid[0:s, 0:s]
        self._xs = xs.astype(np.float32) + 0.5
        self._ys = ys.astype(np.float32) + 0.5
        self.paddle_y = s - self.cfg.paddle_height - 1

    # Ground-truth state: [ball_x, ball_y, ball_vx, ball_vy, paddle_x], all in pixels.
    def state(self) -> np.ndarray:
        return np.array([self.bx, self.by, self.vx, self.vy, self.px], dtype=np.float32)

    def reset(self) -> np.ndarray:
        c, s = self.cfg, self.cfg.size
        self.t = 0
        self.bx = self.rng.uniform(c.ball_radius + 2, s - c.ball_radius - 2)
        self.by = self.rng.uniform(c.ball_radius + 2, s * 0.5)
        speed = self.rng.uniform(*c.ball_speed)
        angle = self.rng.uniform(np.pi * 0.15, np.pi * 0.85)  # mostly downward
        self.vx = speed * np.cos(angle) * self.rng.choice([-1, 1])
        self.vy = speed * np.sin(angle)
        self.px = self.rng.uniform(c.paddle_width / 2, s - c.paddle_width / 2)
        return self.render()

    def step(self, action: int):
        c, s = self.cfg, self.cfg.size
        self.t += 1
        self.px += (action - 1) * c.paddle_speed
        self.px = float(np.clip(self.px, c.paddle_width / 2, s - c.paddle_width / 2))

        self.bx += self.vx
        self.by += self.vy
        r = c.ball_radius
        if self.bx < r:
            self.bx, self.vx = 2 * r - self.bx, -self.vx
        elif self.bx > s - r:
            self.bx, self.vx = 2 * (s - r) - self.bx, -self.vx
        if self.by < r:
            self.by, self.vy = 2 * r - self.by, -self.vy

        reward, done = 0.0, False
        if self.vy > 0 and self.by + r >= self.paddle_y:
            if abs(self.bx - self.px) <= c.paddle_width / 2 + r:
                self.by = 2 * (self.paddle_y - r) - self.by
                self.vy = -self.vy
                # Where the ball hits the paddle nudges its horizontal speed.
                self.vx += 0.5 * (self.bx - self.px) / (c.paddle_width / 2)
                self.vx = float(np.clip(self.vx, -c.ball_speed[1], c.ball_speed[1]))
                reward = 1.0
            elif self.by - r > self.paddle_y:
                reward, done = -1.0, True
        if self.t >= c.max_steps:
            done = True
        return self.render(), reward, done, {"state": self.state()}

    def render(self) -> np.ndarray:
        c = self.cfg
        frame = np.zeros((c.size, c.size), dtype=np.float32)
        # Paddle: a solid rectangle.
        x0 = int(round(self.px - c.paddle_width / 2))
        frame[self.paddle_y : self.paddle_y + c.paddle_height, max(x0, 0) : x0 + c.paddle_width] = PADDLE_VALUE
        # Ball: anti-aliased disc, drawn over the paddle.
        dist = np.sqrt((self._xs - self.bx) ** 2 + (self._ys - self.by) ** 2)
        ball = np.clip(c.ball_radius + 0.5 - dist, 0.0, 1.0) * BALL_VALUE
        return np.maximum(frame, ball)
