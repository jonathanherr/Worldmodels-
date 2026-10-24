"""Collect Squares in Squares datasets for training and held-out evaluation."""

import argparse
from pathlib import Path

import numpy as np

from wm.squares import collect

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=5, help="number of unit squares")
    p.add_argument("--train-steps", type=int, default=100_000)
    p.add_argument("--test-steps", type=int, default=10_000)
    p.add_argument("--out", default="data/squares")
    args = p.parse_args()

    for name, steps, seed in [("train", args.train_steps, 0), ("test", args.test_steps, 12345)]:
        d = collect(steps, n=args.n, seed=seed)
        path = Path(f"{args.out}/n{args.n}_{name}.npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **d)
        lens = np.diff(d["episode_starts"])
        print(f"{name}: {len(d['obs'])} frames, {len(lens)} episodes, mean length {lens.mean():.1f}, "
              f"return per episode {d['reward'].sum() / len(lens):.3f}")
