"""Baseline solver: run the squeeze heuristic in the simulator for each n and keep the best.

For every n it compares the grid packing with a few squeeze episodes, polishes the winner
into a strictly non-overlapping packing, and writes a per-n table and a submission-style CSV.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from wm.squares import (KNOWN_BEST, SUBMISSION_COLUMNS, official_check, SquarePackingEnv, SquaresConfig, grid_solution, polish,
                        squeeze_policy, to_submission_rows, validate)


def solve(n: int, seeds: int, steps: int):
    best = grid_solution(n)
    best_side = validate(*best)["side"]
    for seed in range(seeds):
        env = SquarePackingEnv(SquaresConfig(n=n, max_steps=steps), seed=seed)
        env.reset()
        rng = np.random.default_rng(seed)
        done = False
        while not done:
            _, _, done, _ = env.step(squeeze_policy(env, rng))
        x, y, deg, report = polish(*env.best_solution)
        if report["valid"] and report["side"] < best_side - 1e-6:
            best, best_side = (x, y, deg), report["side"]
    return best


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-min", type=int, default=1)
    p.add_argument("--n-max", type=int, default=12)
    p.add_argument("--seeds", type=int, default=4)
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--out", default="runs/squares")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    solutions, table = {}, []
    for n in range(args.n_min, args.n_max + 1):
        solutions[n] = solve(n, args.seeds, args.steps)
        r = validate(*solutions[n])
        grid = validate(*grid_solution(n))["side"]
        row = {"n": n, "side": r["side"], "score": r["score"], "grid_side": grid, "known_best": KNOWN_BEST.get(n)}
        table.append(row)
        kb = f"{row['known_best']:.4f}" if row["known_best"] else "   -  "
        print(f"n={n:3d}  side {r['side']:.4f}  grid {grid:.0f}  known best {kb}  score {r['score']:.4f}")
    solved = sum(r["score"] for r in table)
    # Missing puzzles score as the grid, so this is what the leaderboard would show for the file.
    rest = sum(np.ceil(np.sqrt(n)) / np.sqrt(n) - 1 for n in range(1, 1001) if n not in solutions)
    print(f"score for n={args.n_min}..{args.n_max}: {solved:.4f}   leaderboard total (grid elsewhere): {solved + rest:.4f}")
    (out / "results.json").write_text(json.dumps(table, indent=1))
    rows = to_submission_rows(solutions)
    # Score every puzzle we solved with Kaggle's own metric, from the values as written out.
    for n in solutions:
        c = official_check([r for r in rows if r["n"] == n])
        assert c["valid"], f"n={n} rejected by the official metric: {c['reason']}"
    with open(out / "submission.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUBMISSION_COLUMNS)
        w.writeheader()
        w.writerows(rows)
