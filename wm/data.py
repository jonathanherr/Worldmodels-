"""Collect episodes with a random policy and serve fixed-length training sequences."""

from pathlib import Path

import numpy as np
import torch

from .env import NUM_ACTIONS, BouncingBallEnv


def sticky_random_policy(rng: np.random.Generator, prev: int, repeat_prob: float = 0.8) -> int:
    """Random actions that tend to repeat, so the paddle actually travels across the screen."""
    if rng.random() < repeat_prob:
        return prev
    return int(rng.integers(NUM_ACTIONS))


def collect(num_steps: int, seed: int = 0) -> dict:
    """Run the random policy until num_steps transitions are gathered.

    Returns flat arrays plus episode boundaries. obs[i] is the frame the agent saw
    before taking action[i]; reward[i] and done[i] are the result of that action.
    The terminal frame of each episode is stored too, so each episode of n actions
    has n + 1 frames (actions/rewards for the extra frame are padding).
    """
    env = BouncingBallEnv(seed=seed)
    rng = np.random.default_rng(seed + 1)
    obs, act, rew, done, state, starts = [], [], [], [], [], []
    while len(obs) < num_steps:
        starts.append(len(obs))
        o, a, d = env.reset(), int(rng.integers(NUM_ACTIONS)), False
        while not d:
            a = sticky_random_policy(rng, a)
            s = env.state()
            o2, r, d, _ = env.step(a)
            obs.append(o); act.append(a); rew.append(r); done.append(d); state.append(s)
            o = o2
        # Terminal frame, with a padding action and a done flag so sequences never cross it.
        obs.append(o); act.append(1); rew.append(0.0); done.append(True); state.append(env.state())
    return {
        "obs": (np.stack(obs) * 255).astype(np.uint8),
        "action": np.array(act, dtype=np.int64),
        "reward": np.array(rew, dtype=np.float32),
        "done": np.array(done, dtype=bool),
        "state": np.stack(state).astype(np.float32),
        "episode_starts": np.array(starts + [len(obs)], dtype=np.int64),
    }


def save(data: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)


def load(path: str | Path) -> dict:
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


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
        return {
            "obs": torch.from_numpy(d["obs"][idx]).float().div_(255).unsqueeze(2).to(device),  # B,T,1,H,W
            "action": torch.from_numpy(d["action"][idx]).to(device),
            "reward": torch.from_numpy(d["reward"][idx]).to(device),
            "done": torch.from_numpy(d["done"][idx].astype(np.float32)).to(device),
            "state": torch.from_numpy(d["state"][idx]).to(device),
        }
