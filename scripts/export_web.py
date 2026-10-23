"""Export the game world model's dream loop for the browser.

The page only needs the dynamics (GRU + prior), the decoder and the heads; it starts from
latent states computed here from real gameplay, so the encoder is not shipped.

Writes to --out:
  weights.bin     float32 tensors, concatenated
  model.json      shapes, offsets, start states, and a reference step for testing
Conv-transpose weights are stored as (in, kH, kW, out) and the decoder's first linear layer
emits a 4x4 grid in height-width-channel order, which is what web/dream.js expects.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.eval_game import load_model
from wm.data import load
from wm.game import ROOM_COLOURS, room_of

ROOM_NAMES = {0: "Meadow", 1: "Desert", 2: "Stone hall", 3: "Cabin"}


def collect_tensors(model):
    t = {}
    sd = {k: v.detach().float() for k, v in model.state_dict().items()}
    for name in ["prior", "reward_head", "continue_head"]:
        t[f"{name}.0.weight"], t[f"{name}.0.bias"] = sd[f"{name}.0.weight"], sd[f"{name}.0.bias"]
        t[f"{name}.2.weight"], t[f"{name}.2.bias"] = sd[f"{name}.2.weight"], sd[f"{name}.2.bias"]
    for k in ["weight_ih", "weight_hh", "bias_ih", "bias_hh"]:
        t[f"gru.{k}"] = sd[f"gru.{k}"]
    W, b = sd["decoder.fc.weight"], sd["decoder.fc.bias"]
    C = W.shape[0] // 16
    t["decoder.fc.weight"] = W.view(C, 4, 4, -1).permute(1, 2, 0, 3).reshape(16 * C, -1)
    t["decoder.fc.bias"] = b.view(C, 4, 4).permute(1, 2, 0).reshape(-1)
    convs = [k for k in sd if k.startswith("decoder.net.") and k.endswith(".weight")]
    for i, k in enumerate(sorted(convs, key=lambda s: int(s.split(".")[2]))):
        t[f"decoder.deconv{i}.weight"] = sd[k].permute(0, 2, 3, 1).contiguous()  # in,kH,kW,out
        t[f"decoder.deconv{i}.bias"] = sd[k.replace("weight", "bias")]
    return t


@torch.no_grad()
def start_states(model, data, context=8, per_room=2, seed=0):
    """Pick real context windows from each room and run the model over them."""
    rng = np.random.default_rng(seed)
    bounds = data["episode_starts"]
    found = {r: [] for r in range(4)}
    for e in rng.permutation(len(bounds) - 1):
        s, end = int(bounds[e]), int(bounds[e + 1])
        if end - s < context + 1:
            continue
        x, y = data["state"][s + context, :2].astype(int)
        r = room_of(x, y)
        if len(found[r]) < per_room:
            found[r].append(s)
        if all(len(v) == per_room for v in found.values()):
            break
    starts = []
    for r, idxs in found.items():
        for s in idxs:
            sl = slice(s, s + context + 1)
            obs = torch.from_numpy(data["obs"][sl]).permute(0, 3, 1, 2).float().div(255)[None]
            act = torch.from_numpy(data["action"][sl])[None]
            z = model.encode(obs)
            z_pred, h = model.rollout(z, act, context)
            starts.append({
                "room": ROOM_NAMES[r], "colour": ROOM_COLOURS[r],
                "h": h[0, -1].tolist(), "z": z_pred[0, -1].tolist(),
            })
    return starts


@torch.no_grad()
def reference_step(model, start):
    """One step from the first start state with action RIGHT, for checking the JS port."""
    h = torch.tensor([start["h"]])
    z = torch.tensor([start["z"]])
    h2 = model.step(h, z, torch.tensor([4]))
    z2 = torch.nn.functional.layer_norm(model.prior(h2), (model.latent_dim,))
    frame = torch.sigmoid(model.decoder(z2))[0].permute(1, 2, 0)  # H,W,C
    return {
        "action": 4, "h": h2[0].tolist(), "z": z2[0].tolist(),
        "frame_sample": frame.flatten()[::97].tolist(),
        "reward_logits": model.reward_head(h2)[0].tolist(),
        "continue_logit": float(model.continue_head(h2)[0, 0]),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="runs/game")
    p.add_argument("--data", default="data/game/test.npz")
    p.add_argument("--out", default="web")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model, ckpt = load_model(Path(args.run))
    tensors = collect_tensors(model)
    manifest, blobs, offset = {}, [], 0
    for name, v in tensors.items():
        arr = v.numpy().astype(np.float32).ravel()
        manifest[name] = {"shape": list(v.shape), "offset": offset, "length": int(arr.size)}
        blobs.append(arr)
        offset += arr.size
    np.concatenate(blobs).tofile(out / "weights.bin")

    starts = start_states(model, load(args.data))
    meta = {
        "latent": model.latent_dim, "hidden": model.hidden_dim, "channels": model.in_ch,
        "num_actions": model.num_actions, "decoder_width": tensors["decoder.fc.bias"].numel() // 16,
        "checkpoint_step": ckpt["step"], "tensors": manifest, "starts": starts,
        "reference": reference_step(model, starts[0]),
    }
    (out / "model.json").write_text(json.dumps(meta))
    print(f"wrote {offset * 4 / 1e6:.1f} MB of weights, {len(starts)} start states")
