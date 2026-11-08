"""Record growth runs as frames for the browser visualization.

Writes JSON {runs: [{label, n, frames: [[x, y, t, s, growing] ...], final: {...}}]} with numbers
rounded for size. Each run's last frame is the jammed state; `final` is it after squeeze.

  python scripts/grow_frames.py --n 27 --out grow_frames.json
"""

import argparse
import json

import numpy as np

from wm.grow import GrowConfig, finish, grow

RUNS = [
    ("All grow together, push", dict(k=0)),
    ("One at a time, push", dict(k=1)),
    ("Groups of 3, independent (no two touching grow)", dict(k=3, adjacency="independent")),
    ("Quarter at a time, one side grows, no push", dict(k=-4, direction="side", adjacency="nopush")),
    ("All, corner growth, local push", dict(k=0, direction="corner", adjacency="local")),
    ("All, open squares grow faster (contacts)", dict(k=0, rate="contacts")),
    ("One at a time, grows toward free side, local push", dict(k=1, direction="free", adjacency="local")),
    ("All, random rates, sizes may differ 2x", dict(k=0, rate="random", band=1.0)),
]


def r4(a):
    return [round(float(v), 4) for v in a]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=27)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--frames", type=int, default=120)
    p.add_argument("--out", default="grow_frames.json")
    a = p.parse_args()
    runs = []
    for label, kw in RUNS:
        kw = dict(kw)
        if kw.get("k", 0) < 0:
            kw["k"] = max(2, a.n // -kw["k"])
        probe = grow(a.n, GrowConfig(seed=a.seed, **kw))
        accepted = max(1, probe["steps"] - probe["fails"])
        r = grow(a.n, GrowConfig(seed=a.seed, record=max(1, accepted // a.frames), **kw))
        fx, fy, ft, fL = finish(r)
        frames = [[r4(x), r4(y), r4(t), r4(s), [int(i) for i in G]] for x, y, t, s, G in r["frames"]]
        frames.append([r4(r["x"]), r4(r["y"]), r4(r["t"]), r4(r["s"]), []])
        runs.append(dict(label=label, n=a.n, steps=r["steps"], side_direct=round(r["side_direct"], 4),
                         frames=frames, final=dict(x=r4(fx / fL), y=r4(fy / fL), t=r4(ft), L=round(float(fL), 5))))
        print(label, r["steps"], round(r["side_direct"], 4), round(float(fL), 4), len(frames), flush=True)
    json.dump({"runs": runs}, open(a.out, "w"), separators=(",", ":"))
