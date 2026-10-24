"""Stand-in solver for tests/test_fanout.py: records which n and seed it was asked to run."""

import argparse
import os
import time
from pathlib import Path

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n")
    p.add_argument("--seed")
    p.add_argument("--workers")
    p.add_argument("--sleep", type=float, default=0.0)
    args = p.parse_args()
    time.sleep(args.sleep)
    out = Path(os.environ["SQUARES_BEST_DIR"]) / "ran"
    out.mkdir(exist_ok=True)
    (out / f"{args.n}_s{args.seed}_{os.getpid()}").touch()
