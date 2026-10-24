"""Write the full 1..1000 submission CSV from the best-per-n store (grid where nothing is stored)."""

import argparse
import csv

from wm import best_store
from wm.squares import SUBMISSION_COLUMNS

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="submission.csv")
    args = p.parse_args()
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUBMISSION_COLUMNS)
        w.writeheader()
        w.writerows(best_store.submission_rows())
    print(f"wrote {args.out}; store total {best_store.total_score():.6f}")
