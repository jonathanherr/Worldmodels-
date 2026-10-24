"""Train the v0 world model on collected random-policy data."""

import argparse
import json
import time
from pathlib import Path

import torch

from wm.data import SequenceSampler, load
from wm.model import WorldModelV0, compute_loss


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/train.npz")
    p.add_argument("--out", default="runs/v0")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--context", type=int, default=8)
    p.add_argument("--ch", type=int, default=32, help="conv width; 16 is fine on CPU")
    p.add_argument("--latent", type=int, default=32)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    data = load(args.data)
    sampler = SequenceSampler(data, args.seq_len, seed=args.seed)
    args.in_ch, args.num_actions = int(data["obs"].shape[-1]), int(data["num_actions"])
    model = WorldModelV0(args.latent, args.hidden, args.ch, args.in_ch, args.num_actions).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    print(f"device={device} params={sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    log, t0 = [], time.time()
    for step in range(1, args.steps + 1):
        batch = sampler.sample(args.batch, device)
        loss, metrics = compute_loss(model, batch, args.context)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
        opt.step()
        if step % 50 == 0 or step == 1:
            metrics.update(step=step, seconds=round(time.time() - t0, 1))
            log.append(metrics)
            print(" ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()), flush=True)
        if step % 500 == 0 or step == args.steps:
            torch.save({"model": model.state_dict(), "args": vars(args), "step": step}, out / "model.pt")
    (out / "train_log.json").write_text(json.dumps(log, indent=1))
