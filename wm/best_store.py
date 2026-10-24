"""Shared best-per-n store for Squares in Squares.

Every solver (neuroevolution, known constructions, the squeeze baseline) offers its packings
here. A packing is kept only if Kaggle's official metric accepts it and its side is shorter
than the stored one, so the store always holds the best certified packing found so far.

Layout: one JSON file per puzzle, `<dir>/n0005.json`:

    {"n": 5, "side": 2.7071067822, "score": 0.2106..., "source": "neuroevo",
     "found": "2026-10-24T11:02:03Z",
     "x": ["0.5", ...], "y": [...], "angle": [...]}

x, y and angle (radians) are the exact decimal strings that go into the submission CSV, so a
stored packing scores the same when it is written out. The directory is `$SQUARES_BEST_DIR`
if set (point several runs at one shared folder), otherwise `solutions/` in the repo.
Writes go through a temp file and a rename, so concurrent writers never leave a torn file, and
the compare-and-replace runs under a per-n lock file, so of two runs storing the same n at
once the shorter packing always wins.
"""

import json
import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from .squares import MAX_N, official_check, puzzle_score

REPO_DIR = Path(__file__).resolve().parent.parent / "solutions"


def store_dir() -> Path:
    return Path(os.environ.get("SQUARES_BEST_DIR", REPO_DIR))


def _path(n: int, root: Path | None = None) -> Path:
    return (root or store_dir()) / f"n{n:04d}.json"


@contextmanager
def _lock(n: int, root: Path | None = None, stale: float = 60.0):
    """Hold `<dir>/n0005.lock` (created exclusively) for the duration; a lock older than `stale`
    seconds belongs to a run that died and is broken."""
    p = _path(n, root).with_suffix(".lock")
    p.parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            os.close(os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            break
        except FileExistsError:
            try:
                if time.time() - p.stat().st_mtime > stale:
                    p.unlink()
            except FileNotFoundError:
                pass
            time.sleep(0.01)
    try:
        yield
    finally:
        p.unlink(missing_ok=True)


def grid_side(n: int) -> float:
    return float(np.ceil(np.sqrt(n)))


def load(n: int, root: Path | None = None) -> dict | None:
    """The stored entry for n, or None if nothing has been stored."""
    p = _path(n, root)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def best_side(n: int, root: Path | None = None) -> float:
    """Side of the stored packing for n, or the grid side if none is stored."""
    e = load(n, root)
    return e["side"] if e else grid_side(n)


def load_xyd(n: int, root: Path | None = None):
    """The stored packing as float arrays (x, y, degrees), or None."""
    e = load(n, root)
    if e is None:
        return None
    return (np.array(e["x"], dtype=float), np.array(e["y"], dtype=float),
            np.rad2deg(np.array(e["angle"], dtype=float)))


def offer(n: int, x, y, deg, source: str, root: Path | None = None, margin: float = 1e-12) -> dict:
    """Store (x, y, deg) for n if the official metric accepts it and it beats the stored side.

    Returns {"accepted": bool, "side": ..., "previous": ..., "reason": ...}.
    """
    x, y, deg = (np.asarray(a, dtype=np.float64) for a in (x, y, deg))
    assert len(x) == n, f"expected {n} squares, got {len(x)}"
    rows = [{"id": k, "n": n, "square_id": k, "x": float(x[k]), "y": float(y[k]),
             "angle": float(np.deg2rad(deg[k]))} for k in range(n)]
    previous = best_side(n, root)
    check = official_check(rows)
    if not check["valid"]:
        return {"accepted": False, "side": None, "previous": previous, "reason": check["reason"]}
    side = check["side"]
    if side >= previous - margin:
        return {"accepted": False, "side": side, "previous": previous, "reason": "not better"}
    entry = {"n": n, "side": side, "score": puzzle_score(side, n), "source": source,
             "found": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "x": [repr(r["x"]) for r in rows], "y": [repr(r["y"]) for r in rows],
             "angle": [repr(r["angle"]) for r in rows]}
    p = _path(n, root)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(f".tmp{os.getpid()}-{uuid.uuid4().hex[:8]}")
    tmp.write_text(json.dumps(entry))
    # Re-read under the lock so a better packing written meanwhile by another run wins.
    with _lock(n, root):
        current = best_side(n, root)
        if side >= current - margin:
            tmp.unlink()
            return {"accepted": False, "side": side, "previous": current, "reason": "beaten meanwhile"}
        os.replace(tmp, p)
    return {"accepted": True, "side": side, "previous": previous, "reason": "ok"}


def submission_rows(root: Path | None = None, max_n: int = MAX_N) -> list[dict]:
    """Full submission rows for n = 1..max_n: the stored packing where there is one, grid elsewhere."""
    rows = []
    for n in range(1, max_n + 1):
        e = load(n, root)
        if e is None:
            k = int(np.ceil(np.sqrt(n)))
            xs = [repr(i % k + 0.5) for i in range(n)]
            ys = [repr(i // k + 0.5) for i in range(n)]
            angs = ["0.0"] * n
        else:
            xs, ys, angs = e["x"], e["y"], e["angle"]
        for k in range(n):
            rows.append({"id": len(rows), "n": n, "square_id": k, "x": xs[k], "y": ys[k], "angle": angs[k]})
    return rows


def total_score(root: Path | None = None, max_n: int = MAX_N) -> float:
    """Leaderboard total implied by the store (grid for puzzles with nothing stored)."""
    return float(sum(puzzle_score(best_side(n, root), n) for n in range(1, max_n + 1)))
