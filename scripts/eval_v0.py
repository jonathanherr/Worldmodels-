"""Evaluate a trained world model on held-out episodes.

Produces, in the run directory:
  rollout_error.png   pixel error and ball-position error vs. imagination horizon
  rollouts.gif        real (left) vs. imagined (right) for a few test sequences
  metrics.json        all numbers, including a linear probe of the latent for true state
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
import numpy as np
import torch

from wm.data import SequenceSampler, load
from wm.env import BALL_VALUE, PADDLE_VALUE
from wm.model import WorldModelV0, class_to_reward

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BALL_THRESHOLD = (BALL_VALUE + PADDLE_VALUE) / 2
STATE_NAMES = ["ball_x", "ball_y", "ball_vx", "ball_vy", "paddle_x"]


def ball_centroid(frames: np.ndarray):
    """frames: N,H,W in [0,1] -> (N,2) xy centroid of ball pixels, and a found mask."""
    w = np.where(frames > BALL_THRESHOLD, frames, 0.0)
    mass = w.sum((1, 2))
    H, W = frames.shape[1:]
    xs, ys = np.arange(W) + 0.5, np.arange(H) + 0.5
    cx = (w.sum(1) * xs).sum(1) / np.maximum(mass, 1e-6)
    cy = (w.sum(2) * ys).sum(1) / np.maximum(mass, 1e-6)
    return np.stack([cx, cy], 1), mass > 0.5


def ridge_r2(x_tr, y_tr, x_te, y_te, alpha=1e-2):
    """Linear probe with a bias term; returns R^2 per target column on the test split."""
    add_bias = lambda x: np.concatenate([x, np.ones((len(x), 1))], 1)  # noqa: E731
    A, B = add_bias(x_tr), add_bias(x_te)
    w = np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ y_tr)
    pred = B @ w
    return 1 - ((pred - y_te) ** 2).sum(0) / ((y_te - y_te.mean(0)) ** 2).sum(0)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="runs/v0")
    p.add_argument("--data", default="data/test.npz")
    p.add_argument("--context", type=int, default=8)
    p.add_argument("--horizon", type=int, default=40)
    p.add_argument("--num", type=int, default=256)
    args = p.parse_args()
    run = Path(args.run)

    ckpt = torch.load(run / "model.pt", map_location="cpu")
    a = ckpt["args"]
    model = WorldModelV0(a.get("latent", 32), a.get("hidden", 256), a.get("ch", 32), a.get("in_ch", 1), a.get("num_actions", 3))
    model.load_state_dict(ckpt["model"])
    model.eval()

    data = load(args.data)
    T = args.context + args.horizon
    batch = SequenceSampler(data, T, seed=0).sample(args.num)
    obs, act, state = batch["obs"], batch["action"], batch["state"]

    z = model.encode(obs)
    z_open, h_open = model.rollout(z, act, args.context)     # imagination after the context
    z_tf, h_tf = model.rollout(z, act, T)                    # teacher forced throughout
    pred = torch.sigmoid(model.decode(z_open)).squeeze(2).numpy()  # N,T-1,H,W = frames 1..T-1
    real = obs.squeeze(2).numpy()
    one_step = torch.sigmoid(model.decode(z_tf)).squeeze(2).numpy()

    # ---- Error vs. horizon (open-loop frames only) ----
    k0 = args.context  # first open-loop frame index
    horizons = np.arange(1, args.horizon + 1)
    pix_mse, pix_copy = [], []
    pos_err, pos_copy, lost = [], [], []
    last_ctx = real[:, k0 - 1]
    true_xy = state[..., :2].numpy()
    for k in horizons:
        t = k0 + k - 1
        pix_mse.append(float(((pred[:, t - 1] - real[:, t]) ** 2).mean()))
        pix_copy.append(float(((last_ctx - real[:, t]) ** 2).mean()))
        xy, found = ball_centroid(pred[:, t - 1])
        err = np.linalg.norm(xy - true_xy[:, t], axis=1)
        pos_err.append(float(np.median(err[found])) if found.any() else float("nan"))
        lost.append(float(1 - found.mean()))
        pos_copy.append(float(np.median(np.linalg.norm(true_xy[:, k0 - 1] - true_xy[:, t], axis=1))))

    one_step_mse = float(((one_step[:, :-1] - real[:, 1:-1]) ** 2).mean())
    recon_mse = float(((torch.sigmoid(model.decode(z)).squeeze(2).numpy() - real) ** 2).mean())

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(horizons, pix_mse, label="world model")
    ax[0].plot(horizons, pix_copy, "--", label="copy last frame")
    ax[0].set(xlabel="steps imagined after context", ylabel="pixel MSE", title="Frame error vs horizon")
    ax[1].plot(horizons, pos_err, label="world model")
    ax[1].plot(horizons, pos_copy, "--", label="ball stays put")
    ax[1].set(xlabel="steps imagined after context", ylabel="median ball position error (px)",
              title="Ball position error vs horizon")
    for a in ax:
        a.legend(); a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(run / "rollout_error.png", dpi=120)

    # ---- Reward and termination heads, one step, teacher forced ----
    r_true = batch["reward"][:, :-1].flatten()
    r_pred = class_to_reward(model.reward_head(h_tf).argmax(-1)).flatten()
    reward_metrics = {}
    for name, v in [("hit", 1.0), ("miss", -1.0)]:
        m = r_true == v
        reward_metrics[f"{name}_recall"] = float((r_pred[m] == v).float().mean()) if m.any() else None
        pm = r_pred == v
        reward_metrics[f"{name}_precision"] = float((r_true[pm] == v).float().mean()) if pm.any() else None
    d_true = batch["done"][:, :-1].flatten() > 0.5
    d_pred = torch.sigmoid(model.continue_head(h_tf).squeeze(-1)).flatten() < 0.5
    reward_metrics["done_accuracy_on_terminal_steps"] = float(d_pred[d_true].float().mean()) if d_true.any() else None

    # ---- Linear probes: does the latent encode the true state? ----
    s = state[:, 1:].reshape(-1, 5).numpy()
    feats = {"z (encoder, one frame)": z[:, 1:].reshape(-1, z.shape[-1]).numpy(),
             "h (GRU memory)": h_tf.reshape(-1, h_tf.shape[-1]).numpy()}
    half = (args.num // 2) * (T - 1)
    probes = {name: dict(zip(STATE_NAMES, map(float, ridge_r2(f[:half], s[:half], f[half:], s[half:]))))
              for name, f in feats.items()}

    # ---- Video: real | imagined, context frames outlined in grey, imagined in orange ----
    frames_out, scale, n_show = [], 3, 4
    for t in range(1, T):
        rows = []
        for i in range(n_show):
            r = np.repeat(real[i, t][..., None], 3, -1)
            im = np.repeat((real[i, t] if t < k0 else pred[i, t - 1])[..., None], 3, -1)
            border = np.array([0.5, 0.5, 0.5]) if t < k0 else np.array([1.0, 0.55, 0.1])
            im = np.pad(im, ((2, 2), (2, 2), (0, 0)), constant_values=0)
            im[:2], im[-2:], im[:, :2], im[:, -2:] = border, border, border, border
            r = np.pad(r, ((2, 2), (2, 2), (0, 0)), constant_values=0.25)
            rows.append(np.concatenate([r, np.full((68, 4, 3), 1.0), im], 1))
            rows.append(np.full((4, rows[-1].shape[1], 3), 1.0))
        img = np.concatenate(rows, 0)
        img = np.kron(img, np.ones((scale, scale, 1)))
        frames_out.append((img * 255).astype(np.uint8))
    imageio.mimsave(run / "rollouts.gif", frames_out, duration=0.1, loop=0)

    metrics = {
        "checkpoint_step": ckpt["step"], "context": args.context, "horizon": args.horizon,
        "num_sequences": args.num, "recon_mse": recon_mse, "one_step_mse": one_step_mse,
        "rollout_pixel_mse": dict(zip(map(int, horizons), pix_mse)),
        "copy_last_frame_pixel_mse": dict(zip(map(int, horizons), pix_copy)),
        "rollout_ball_pos_err_px_median": dict(zip(map(int, horizons), pos_err)),
        "ball_stays_put_err_px_median": dict(zip(map(int, horizons), pos_copy)),
        "ball_lost_rate": dict(zip(map(int, horizons), lost)),
        "heads": reward_metrics, "probe_r2": probes,
    }
    (run / "metrics.json").write_text(json.dumps(metrics, indent=1))
    for k in [1, 5, 10, 20, 40]:
        if k <= args.horizon:
            print(f"h={k:2d} pixMSE={pix_mse[k-1]:.5f} (copy {pix_copy[k-1]:.5f}) "
                  f"ballErr={pos_err[k-1]:.2f}px (static {pos_copy[k-1]:.2f}) lost={lost[k-1]:.2f}")
    print("recon", recon_mse, "one-step", one_step_mse)
    print(json.dumps(reward_metrics, indent=1))
    print(json.dumps(probes, indent=1))


if __name__ == "__main__":
    main()
