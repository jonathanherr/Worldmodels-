"""Collect random-policy datasets for training and held-out evaluation."""

import argparse

import numpy as np

from wm.data import collect, save

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train-steps", type=int, default=100_000)
    p.add_argument("--test-steps", type=int, default=10_000)
    p.add_argument("--env", choices=["ball", "game"], default="ball")
    p.add_argument("--out", default=None, help="default: data/ for ball, data/game/ for game")
    args = p.parse_args()
    out = args.out or ("data" if args.env == "ball" else f"data/{args.env}")

    for name, steps, seed in [("train", args.train_steps, 0), ("test", args.test_steps, 12345)]:
        d = collect(steps, seed=seed, env_name=args.env)
        save(d, f"{out}/{name}.npz")
        lens = np.diff(d["episode_starts"])
        print(f"{name}: {len(d['obs'])} frames, {len(lens)} episodes, mean length {lens.mean():.1f}, "
              f"hits {(d['reward'] > 0).sum()}, misses {(d['reward'] < 0).sum()}")
