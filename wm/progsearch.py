"""Program search: agents write small Python programs that rebuild one window of a record packing.

The open problem "pack n squares better" is cut down to one fixed, scoreable task:

* A task names one or more n (a family that shares the same structure, such as the diagonal-strip
  packings 299, 335, 412), a corner of the box and a window size. The record packing for each n is
  loaded; the k squares whose centres lie in the size x size window at that corner are the
  program's to rearrange, everything else stays where it is.
* A program is a file defining `build(ctx) -> list of (x, y, angle_deg)` with exactly ctx["k"]
  entries. ctx is given in window-local coordinates: the corner is (0, 0), the two box walls that
  meet there are x = 0 and y = 0, and the window is [0, size]^2. ctx also lists the frozen
  neighbours around the window, the record's own window layout and the box side. The helper
  library `wm.progsearch_lib` (blocks, rows, rotations, overlap check) is importable.
* The program runs in a subprocess with a CPU limit. Its squares are mapped back into the box
  next to the frozen ones and the whole packing goes through `pack_opt.squeeze` starting at the
  record side, with every square free. The score per n is record side - final side (positive is
  a win; the record layout itself scores about 0). A program only has to be roughly right: the
  squeeze takes care of the last 1e-9.
* Programs and their scores live on a board directory ($PROGSEARCH_BOARD) as plain files, so
  several agents (Claude sessions) can read the leaderboard, copy the best programs and post
  mutations.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from . import best_store
from .pack_opt import MARGIN, compact_squeeze, feasible, squeeze

REPO = Path(__file__).resolve().parent.parent
SEED_DIR = Path(os.environ.get("PROGSEARCH_SEEDS", "/mnt/project-files/squares/public-certs/store"))
BOARD = Path(os.environ.get("PROGSEARCH_BOARD", "/mnt/project-files/squares/progsearch"))

# Kaggle per-puzzle bests that beat the public records (Couzo upper_bounds.json, "reported").
TARGETS = {147: 12.651043553627, 299: 17.802718854399, 335: 18.794904187687, 412: 20.798165581954}

TASKS = {
    # Whole-construction tasks: the program builds all n squares from scratch (it may start from the
    # record in ctx["original"]). Pilot 2 showed the 299/335/412 strip is a local optimum of its own
    # design (g = (sqrt7 - 1)/2 from dominoes spanning one staircase step), so a win needs a new
    # design, staircases included; scoring one program on the whole family keeps it general.
    "construct-strip": {"ns": [299, 335, 412], "select": "all", "corner": "bl", "size": None,
                        "goal": "Build the whole packing of n squares (a construction that works for every n "
                                "in the family). The record is a two-wide strip at 24.3 degrees between two "
                                "axis-aligned staircases, provably optimal for that design; beat it with a "
                                "different design."},
    "construct147": {"ns": [147], "select": "all", "corner": "bl", "size": None,
                     "goal": "Build the whole packing of 147 squares. The record is the 148 layout (45-degree "
                             "8x8 diamond in an axis frame) minus one corner square; beat it with a different design."},
    # Whole-block tasks: the tilted block is the program's, the axis-aligned frame stays. Deleting
    # every tilted square lets the rest compact to 12.0 / 18.0, while deleting any corner, end or
    # half of the block frees nothing (each frame row and column across the block holds the side
    # on its own), so these are the smallest windows found with real headroom.
    "diamond147": {"ns": [147], "select": "tilted", "corner": "bl", "size": None,
                   "goal": "Your squares replace the 64-square 45-degree diamond; the axis-aligned frame "
                           "around it stays. Fill the hole with 64 squares so the whole packing can shrink."},
    "strip": {"ns": [299, 335, 412], "select": "tilted", "corner": "bl", "size": None,
              "goal": "Your squares replace the tilted diagonal strip (and its odd-angle end pieces); the "
                      "axis-aligned staircases on both sides stay. Rebuild the strip so the packing can shrink."},
    # Corner windows: kept as negative controls. With the window emptied the side does not move.
    # Warm-up with a known answer: 147 is the 148 record minus its top-right corner square.
    "corner147": {"ns": [147], "corner": "tr", "size": 4.0,
                  "goal": "The window holds the empty corner cell left by removing square 148. "
                          "Use that hole: rearrange the corner so the whole packing can shrink."},
    # The diagonal-strip family: the strip ends hold about 6.7 of 335's 19.3 units of empty area.
    "strip-bl": {"ns": [299, 335, 412], "corner": "bl", "size": 4.0,
                 "goal": "The window is where the tilted two-wide strip meets the bottom-left corner. "
                         "Rebuild this strip end with less wasted area."},
    "strip-tr": {"ns": [299, 335, 412], "corner": "tr", "size": 4.0,
                 "goal": "The window is where the tilted two-wide strip meets the top-right corner. "
                         "Rebuild this strip end with less wasted area."},
}


def load_seed(n: int, seed_dir: Path = SEED_DIR):
    e = json.loads((Path(seed_dir) / f"n{n:04d}.json").read_text())
    x, y, t = (np.array([float(v) for v in e[k]]) for k in ("x", "y", "angle"))
    return x, y, t, float(e["side"])


# Window-local frame: reflect so the chosen corner is the origin and the window is [0, size]^2.
def _flip(corner: str) -> tuple[bool, bool]:
    return corner in ("tr", "br"), corner in ("tr", "tl")


def to_local(x, y, deg, L, corner):
    fx, fy = _flip(corner)
    lx = L - x if fx else x
    ly = L - y if fy else y
    ld = -deg if fx != fy else deg  # one reflection reverses the sense of rotation
    return lx, ly, ld


def to_global(lx, ly, ld, L, corner):
    return to_local(lx, ly, ld, L, corner)  # the reflection is its own inverse


def make_ctx(n: int, corner: str, size: float | None, seed_dir: Path = SEED_DIR, goal: str = "",
             select: str = "window") -> dict:
    """select="window": squares centred in the size x size corner window; "tilted": every square not
    axis-aligned (size becomes the box side and the frame is the box's own, bl corner)."""
    x, y, t, L = load_seed(n, seed_dir)
    lx, ly, ld = to_local(x, y, np.degrees(t), L, corner)
    if select == "all":
        inside = np.ones(n, dtype=bool)
        size = L
        near = ~inside
    elif select == "tilted":
        a = np.degrees(t) % 90.0
        inside = (a > 0.1) & (a < 89.9)
        size = L
        near = ~inside
    else:
        inside = (lx < size) & (ly < size)
        near = ~inside & (lx < size + 2.0) & (ly < size + 2.0)
    sq = lambda m: [[round(float(a), 9), round(float(b), 9), round(float(c) % 90.0, 6)]
                    for a, b, c in zip(lx[m], ly[m], ld[m])]
    return {"n": n, "corner": corner, "size": size, "side": L, "k": int(inside.sum()),
            "target_side": TARGETS.get(n), "goal": goal,
            "original": sq(inside), "frozen": sq(near), "_inside": inside.tolist()}


RUNNER = r"""
import importlib.util, json, resource, sys
cpu = int(sys.argv[3])
resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
sys.path.insert(0, sys.argv[4])
spec = importlib.util.spec_from_file_location("prog", sys.argv[1])
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
ctx = json.load(open(sys.argv[2]))
out = [[float(a), float(b), float(c)] for a, b, c in mod.build(ctx)]
print("RESULT" + json.dumps(out))
"""


def run_program(path: Path, ctx: dict, cpu_seconds: int = 60) -> list:
    """Run build(ctx) from the program file in a subprocess; returns its (x, y, deg) list."""
    tmp = Path(os.environ.get("TMPDIR", "/tmp")) / f"progsearch_ctx_{os.getpid()}_{ctx['n']}.json"
    tmp.write_text(json.dumps({k: v for k, v in ctx.items() if not k.startswith("_")}))
    try:
        p = subprocess.run([sys.executable, "-I", "-c", RUNNER, str(Path(path).resolve()), str(tmp), str(cpu_seconds), str(REPO)],
                           capture_output=True, text=True, timeout=cpu_seconds * 2 + 10, cwd=tmp.parent)
    finally:
        tmp.unlink(missing_ok=True)
    line = next((s for s in p.stdout.splitlines() if s.startswith("RESULT")), None)
    if line is None:
        raise RuntimeError((p.stderr or p.stdout).strip()[-2000:] or f"exit code {p.returncode}")
    return json.loads(line[6:])


def assemble(ctx: dict, squares: list, seed_dir: Path = SEED_DIR):
    """Global x, y, t (radians) with the window replaced by the program's squares."""
    if len(squares) != ctx["k"]:
        raise ValueError(f"build() returned {len(squares)} squares, the window needs exactly {ctx['k']}")
    x, y, t, L = load_seed(ctx["n"], seed_dir)
    keep = ~np.array(ctx["_inside"])
    a = np.array(squares, dtype=float)
    if not keep.any():  # whole construction: move it to the origin, the side is its own extent
        w = 0.5 * (np.abs(np.cos(np.radians(a[:, 2]))) + np.abs(np.sin(np.radians(a[:, 2]))))
        a[:, 0] -= (a[:, 0] - w).min()
        a[:, 1] -= (a[:, 1] - w).min()
    gx, gy, gd = to_global(a[:, 0], a[:, 1], a[:, 2], L, ctx["corner"])
    return (np.concatenate([x[keep], gx]), np.concatenate([y[keep], gy]),
            np.concatenate([t[keep], np.radians(gd)]), L)


def score_layout(x, y, t, L0, rounds: int = 60, start: float | None = None) -> dict:
    """Side the assembled packing reaches: the better of two shrink routes.

    `squeeze` (scale and relax from the record side) keeps record precision but cannot
    propagate a rearrangement through a jammed packing: with all 45 strip squares of 335 deleted
    it still stops at the record side. `compact_squeeze` makes the side a variable, so it finds
    room freed anywhere (335 minus its strip compacts to about 18.0), at about 1e-3 precision.
    """
    best = None
    Ls = L0 if start is None else start
    for route in ("squeeze", "compact"):
        if route == "squeeze":
            out = squeeze(x, y, t, Ls, rounds=rounds, step=2e-4, min_step=1e-9, maxiter=4000)
        else:
            out = compact_squeeze(x, y, t, Ls * 1.002)
        if out is None:
            continue
        sx, sy, st, sL = out
        if feasible(sx, sy, st, sL, MARGIN) and (best is None or sL < best[3]):
            best = (sx, sy, st, float(sL), route)
    if best is None:
        return {"ok": False, "side": math.inf, "gain": -math.inf}
    sx, sy, st, sL, route = best
    return {"ok": True, "side": sL, "gain": float(L0 - sL), "route": route, "packing": (sx, sy, st)}


def evaluate(path: Path, task: str, seed_dir: Path = SEED_DIR, cpu_seconds: int = 60,
             offer_root: Path | None = None) -> dict:
    """Score one program on every n of a task. Wins are offered to the store at offer_root, if given."""
    spec = TASKS[task]
    per_n, total = {}, 0.0
    for n in spec["ns"]:
        ctx = make_ctx(n, spec["corner"], spec["size"], seed_dir, spec["goal"], spec.get("select", "window"))
        t0 = time.time()
        try:
            sq = run_program(Path(path), ctx, cpu_seconds)
            x, y, t, L0 = assemble(ctx, sq, seed_dir)
            start = None
            if spec.get("select") == "all":  # start from the construction's own enclosing side
                w = 0.5 * (np.abs(np.cos(t)) + np.abs(np.sin(t)))
                start = max(float((x + w).max()), float((y + w).max())) * (1 + 1e-9)
            r = score_layout(x, y, t, L0, start=start)
        except Exception as ex:  # a broken program scores as a failure, with its error shown
            per_n[n] = {"ok": False, "error": str(ex)[-600:]}
            total -= 1.0
            continue
        res = {"ok": r["ok"], "side": r["side"], "gain": r["gain"], "route": r.get("route"), "seconds": round(time.time() - t0, 1)}
        if r["ok"] and r["gain"] > 1e-9 and offer_root is not None:
            px, py, pt = r["packing"]
            res["offer"] = best_store.offer(n, px, py, np.degrees(pt), f"progsearch:{task}", root=offer_root)
        per_n[n] = res
        total += r["gain"] if r["ok"] else -1.0
    return {"task": task, "score": total, "per_n": per_n}


# ---------------- board: programs and scores as files
def _pid(code: str) -> str:
    return hashlib.sha1(code.encode()).hexdigest()[:10]


def submit(path: Path, task: str, author: str, note: str = "", parent: str | None = None,
           board: Path = BOARD, **kw) -> dict:
    code = Path(path).read_text()
    pid = _pid(code)
    d = Path(board) / task
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{pid}.py").write_text(code)
    r = evaluate(d / f"{pid}.py", task, **kw)
    r.update(id=pid, author=author, note=note, parent=parent, time=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    (d / f"{pid}.json").write_text(json.dumps(r, indent=1, default=str))
    return r


def leaderboard(task: str, board: Path = BOARD, top: int = 10) -> list[dict]:
    d = Path(board) / task
    rows = [json.loads(p.read_text()) for p in d.glob("*.json")] if d.exists() else []
    return sorted(rows, key=lambda r: -r["score"])[:top]


def headroom(task: str, seed_dir: Path = SEED_DIR) -> dict:
    """Upper bound on a task's gain: squeeze each packing with the window's squares deleted.

    If the side does not shrink even with the window empty, no program for that window can win
    (the binding contact chain runs elsewhere), so check this before spending agent time.
    """
    spec = TASKS[task]
    if spec.get("select") == "all":
        return {n: {"headroom": "whole construction: unbounded"} for n in spec["ns"]}
    out = {}
    for n in spec["ns"]:
        ctx = make_ctx(n, spec["corner"], spec["size"], seed_dir, select=spec.get("select", "window"))
        x, y, t, L0 = load_seed(n, seed_dir)
        keep = ~np.array(ctx["_inside"])
        r = score_layout(x[keep], y[keep], t[keep], L0)
        out[n] = {"k": ctx["k"], "empty_window_side": r["side"], "headroom": r["gain"],
                  "needed": None if n not in TARGETS else L0 - TARGETS[n]}
    return out
