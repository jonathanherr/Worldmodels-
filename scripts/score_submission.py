"""Score a submission CSV locally with Kaggle's official metric (all puzzles n = 1..1000)."""

import argparse

import pandas as pd

from wm import kaggle_metric

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("csv", help="submission file with columns id,n,square_id,x,y,angle")
    args = p.parse_args()

    submission = pd.read_csv(args.csv, dtype=str)  # keep the decimal strings exactly as written
    solution = pd.DataFrame({"id": range(kaggle_metric.MAX_N), "n": range(1, kaggle_metric.MAX_N + 1)})
    print(f"score {kaggle_metric.score(solution, submission, 'id'):.6f}  (per-puzzle L in submission_details.csv)")
