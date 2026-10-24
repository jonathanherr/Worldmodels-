"""Collect episodes with an exploring policy and serve fixed-length training sequences."""

from pathlib import Path

import numpy as np
import torch

from . import env as ball
from . import game


def sticky_random_policy(rng: np.random.Generator, prev: int, repeat_prob: float = 0.8) -> int:
    """Random actions that tend to repeat, so the paddle actually travels across the screen."""
    if rng.random() < repeat_prob:
        return prev
    return int(rng.integers(ball.NUM_ACTIONS))


ENVS = {
    # name: (constructor, policy(env, rng, prev_action) -> action, number of actions)
    "ball": (ball.BouncingBallEnv, lambda env, rng, prev: sticky_random_policy(rng, prev), ball.NUM_ACTIONS),
    "game": (game.GameWorld, game.explore_policy, game.NUM_ACTIONS),
}


def to_uint8(frame: np.ndarray) -> np.ndarray:
    return (frame * 255).round().astype(np.uint8)


def collect(num_steps: int, seed: int = 0, env_name: str = "ball") -> dict:
    """Run the exploring policy until num_steps transitions are gathered.

    Returns flat arrays plus episode boundaries. obs[i] is the frame the agent saw
    before taking action[i]; reward[i] and done[i] are the result of that action.
    The terminal frame of each episode is stored too, so each episode of n actions
    has n + 1 frames (actions/rewards for the extra frame are padding).
    Frames are stored as uint8 (N, H, W, C).
    """
    make_env, policy, num_actions = ENVS[env_name]
    env = make_env(seed=seed)
    rng = np.random.default_rng(seed + 1)
    obs, act, rew, done, state, mask, starts = [], [], [], [], [], [], []
    while len(obs) < num_steps:
        starts.append(len(obs))
        o, a, d = env.reset(), int(rng.integers(num_actions)), False
        m = env.render(with_mask=True)[1] if env_name == "game" else np.zeros(o.shape[:2], dtype=bool)
        while not d:
            a = policy(env, rng, a)
            s = env.state()
            o2, r, d, info = env.step(a)
            obs.append(to_uint8(o)); act.append(a); rew.append(r); done.append(d); state.append(s); mask.append(m)
            o, m = o2, info.get("sprite_mask", m)
        # Terminal frame, with a padding action and a done flag so sequences never cross it.
        obs.append(to_uint8(o)); act.append(0); rew.append(0.0); done.append(True); state.append(env.state()); mask.append(m)
    obs = np.stack(obs)
    if obs.ndim == 3:
        obs = obs[..., None]
    data = {
        "obs": obs,
        "action": np.array(act, dtype=np.int64),
        "reward": np.array(rew, dtype=np.float32),
        "done": np.array(done, dtype=bool),
        "state": np.stack(state).astype(np.float32),
        "episode_starts": np.array(starts + [len(obs)], dtype=np.int64),
        "num_actions": np.array(num_actions),
    }
    if env_name == "game":
        # Sprite pixels (coins, slimes, player), used only to weight the reconstruction loss.
        data["mask"] = np.stack(mask)
    return data


def save(data: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)


def load(path: str | Path) -> dict:
    with np.load(path) as f:
        d = {k: f[k] for k in f.files}
    if d["obs"].ndim == 3:  # datasets saved before frames had a channel axis
        d["obs"] = d["obs"][..., None]
    d.setdefault("num_actions", np.array(ball.NUM_ACTIONS))
    return d


class SequenceSampler:
    """Samples random windows of seq_len steps that stay inside one episode."""

    def __init__(self, data: dict, seq_len: int, seed: int = 0):
        self.data = data
        self.seq_len = seq_len
        self.rng = np.random.default_rng(seed)
        bounds = data["episode_starts"]
        valid = []
        for start, end in zip(bounds[:-1], bounds[1:]):
            if end - start >= seq_len:
                valid.append(np.arange(start, end - seq_len + 1))
        self.valid_starts = np.concatenate(valid)

    def sample(self, batch_size: int, device="cpu") -> dict:
        starts = self.rng.choice(self.valid_starts, size=batch_size)
        idx = starts[:, None] + np.arange(self.seq_len)[None, :]
        d = self.data
        batch = {
            "obs": torch.from_numpy(d["obs"][idx]).permute(0, 1, 4, 2, 3).float().div_(255).to(device),  # B,T,C,H,W
            "action": torch.from_numpy(d["action"][idx]).to(device),
            "reward": torch.from_numpy(d["reward"][idx]).to(device),
            "done": torch.from_numpy(d["done"][idx].astype(np.float32)).to(device),
            "state": torch.from_numpy(d["state"][idx]).to(device),
        }
        if "mask" in d:
            batch["mask"] = torch.from_numpy(d["mask"][idx]).unsqueeze(2).float().to(device)  # B,T,1,H,W
        return batch
