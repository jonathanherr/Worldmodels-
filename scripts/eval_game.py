"""Evaluate a world model trained on the game world.

Writes to the run directory:
  rollout_error.png   frame error and "where am I" error vs. imagination horizon
  rollouts.gif        real (left) vs. dreamed (right) gameplay for a few test sequences
  long_dream.gif      one long dream (160 steps) driven by the real actions
  metrics.json        the numbers behind the plots
"""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
import numpy as np
import torch

from wm.data import SequenceSampler, load
from wm.model import WorldModelV0

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_model(run: Path) -> tuple[WorldModelV0, dict]:
    ckpt = torch.load(run / "model.pt", map_location="cpu")
    a = ckpt["args"]
    model = WorldModelV0(a["latent"], a["hidden"], a["ch"], a["in_ch"], a["num_actions"])
    model.load_state_dict(ckpt["model"])
    return model.eval(), ckpt


def fit_probe(x, y, alpha=1e-2):
    A = np.concatenate([x, np.ones((len(x), 1))], 1)
    return np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ y)


def apply_probe(w, x):
    return np.concatenate([x, np.ones((len(x), 1))], 1) @ w


def side_by_side(real, dream, k0, scale=3, label_rows=4):
    """real, dream: N,T,H,W,3 in [0,1]; dream frames before k0 are real context."""
    frames = []
    for t in range(real.shape[1]):
        rows = []
        for i in range(min(label_rows, real.shape[0])):
            border = (0.5, 0.5, 0.5) if t < k0 else (1.0, 0.55, 0.1)
            r = np.pad(real[i, t], ((2, 2), (2, 2), (0, 0)), constant_values=0.2)
            d = np.pad(dream[i, t], ((2, 2), (2, 2), (0, 0)))
            d[:2], d[-2:], d[:, :2], d[:, -2:] = border, border, border, border
            rows.append(np.concatenate([r, np.ones((68, 4, 3)), d], 1))
            rows.append(np.ones((4, rows[-1].shape[1], 3)))
        img = np.kron(np.concatenate(rows, 0), np.ones((scale, scale, 1)))
        frames.append((np.clip(img, 0, 1) * 255).astype(np.uint8))
    return frames


@torch.no_grad()
def dream(model, obs, act, context):
    z = model.encode(obs)
    z_pred, h = model.rollout(z, act, context)
    frames = torch.sigmoid(model.decode(z_pred))
    return z, z_pred, h, frames


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="runs/game")
    p.add_argument("--data", default="data/game/test.npz")
    p.add_argument("--context", type=int, default=8)
    p.add_argument("--horizon", type=int, default=48)
    p.add_argument("--num", type=int, default=128)
    args = p.parse_args()
    run = Path(args.run)
    model, ckpt = load_model(run)
    data = load(args.data)
    k0, T = args.context, args.context + args.horizon

    batch = SequenceSampler(data, T, seed=0).sample(args.num)
    obs, act, state = batch["obs"], batch["action"], batch["state"].numpy()
    z, z_pred, h, pred = dream(model, obs, act, k0)
    real = obs.permute(0, 1, 3, 4, 2).numpy()            # N,T,H,W,3
    pred = pred.permute(0, 1, 3, 4, 2).numpy()           # N,T-1,H,W,3 (frames 1..T-1)

    # A linear probe from the encoder latent of REAL frames to the player's map position,
    # applied to DREAMED latents: does the dream still know where the player is?
    zr = z.flatten(0, 1).numpy()
    xy = state[..., :2].reshape(-1, 2)
    half = len(zr) // 2
    w = fit_probe(zr[:half], xy[:half])
    probe_err_real = float(np.abs(apply_probe(w, zr[half:]) - xy[half:]).sum(1).mean())

    horizons = np.arange(1, args.horizon + 1)
    pix, copy, pos_err, pos_ok = [], [], [], []
    for k in horizons:
        t = k0 + k - 1
        pix.append(float(((pred[:, t - 1] - real[:, t]) ** 2).mean()))
        copy.append(float(((real[:, k0 - 1] - real[:, t]) ** 2).mean()))
        est = apply_probe(w, z_pred[:, t - 1].numpy())
        err = np.abs(np.round(est) - state[:, t, :2]).sum(1)  # Manhattan distance in tiles
        pos_err.append(float(err.mean()))
        pos_ok.append(float((err == 0).mean()))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(horizons, pix, label="world model")
    ax[0].plot(horizons, copy, "--", label="copy last frame")
    ax[0].set(xlabel="steps dreamed after context", ylabel="pixel MSE", title="Frame error vs horizon")
    ax[1].plot(horizons, pos_ok, label="dream knows exact tile")
    ax[1].set(xlabel="steps dreamed after context", ylabel="fraction of rollouts", ylim=(0, 1.02),
              title="Player position read from the dream")
    for a in ax:
        a.legend(); a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(run / "rollout_error.png", dpi=120)

    ctx = np.concatenate([real[:, :1], real[:, 1:k0], pred[:, k0 - 1 :]], 1)  # context frames shown as real
    imageio.mimsave(run / "rollouts.gif", side_by_side(real, ctx, k0), duration=0.15, loop=0)

    # One long dream, driven by a long real episode's actions.
    bounds = data["episode_starts"]
    longest = int(np.argmax(np.diff(bounds)))
    s, L = int(bounds[longest]), min(168, int(np.diff(bounds)[longest]))
    o = torch.from_numpy(data["obs"][s : s + L]).permute(0, 3, 1, 2).float().div(255)[None]
    a = torch.from_numpy(data["action"][s : s + L])[None]
    _, _, _, lp = dream(model, o, a, k0)
    lr = o.permute(0, 1, 3, 4, 2).numpy()
    lp = lp.permute(0, 1, 3, 4, 2).numpy()
    ld = np.concatenate([lr[:, :k0], lp[:, k0 - 1 :]], 1)
    imageio.mimsave(run / "long_dream.gif", side_by_side(lr, ld, k0, label_rows=1), duration=0.12, loop=0)
    long_mse = float(((lp[0, k0 - 1 :] - lr[0, k0:]) ** 2).mean())

    metrics = {
        "checkpoint_step": ckpt["step"], "context": k0, "horizon": args.horizon, "num_sequences": args.num,
        "probe_abs_err_tiles_on_real_frames": probe_err_real,
        "rollout_pixel_mse": dict(zip(map(int, horizons), pix)),
        "copy_last_frame_pixel_mse": dict(zip(map(int, horizons), copy)),
        "dream_position_err_tiles": dict(zip(map(int, horizons), pos_err)),
        "dream_position_exact": dict(zip(map(int, horizons), pos_ok)),
        "long_dream_steps": L - k0, "long_dream_pixel_mse": long_mse,
    }
    (run / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print(f"probe error on real frames: {probe_err_real:.2f} tiles")
    for k in [1, 4, 8, 16, 32, 48]:
        if k <= args.horizon:
            print(f"h={k:2d} pixMSE={pix[k-1]:.4f} (copy {copy[k-1]:.4f}) pos_exact={pos_ok[k-1]:.2f} pos_err={pos_err[k-1]:.2f}")
    print(f"long dream: {L - k0} steps, pixel MSE {long_mse:.4f}")


if __name__ == "__main__":
    main()
