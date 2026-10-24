"""Checks for the fan-out runner and the store's per-n lock."""

import json
import os
import threading
import time

import numpy as np

from scripts.fanout import Leases, make_jobs, run
from wm import best_store
from wm.squares import five_solution, polish


def test_jobs_go_worst_score_first_and_cover_every_seed(tmp_path):
    x, y, deg, _ = polish(*five_solution(), gap=1e-9)
    assert best_store.offer(5, x, y, deg, "test", root=tmp_path)["accepted"]
    jobs = make_jobs(range(2, 6), 2, [0, 1], root=tmp_path)
    # n = 2, 3 sit at the grid (2 / sqrt2 - 1 = 0.41 ...), n = 4, 5 are optimal or close to it.
    assert [j["id"] for j in jobs] == ["n0002-0003_s0", "n0004-0005_s0", "n0002-0003_s1", "n0004-0005_s1"]


def test_a_job_is_claimed_once_and_a_stale_lease_is_taken_over(tmp_path):
    job = {"id": "n0010-0010_s0", "ns": [10], "seed": 0}
    a, b = Leases(tmp_path, ttl=0.5), Leases(tmp_path, ttl=0.5)
    b.owner = "other:1"
    assert a.claim(job) and not b.claim(job)
    time.sleep(0.6)
    assert b.claim(job) and not a.renew(job["id"])
    b.finish(job["id"], "done")
    assert not a.claim(job) and a.claim(job, rerun=True)


def test_two_runners_on_one_store_split_the_jobs(tmp_path, monkeypatch):
    monkeypatch.setenv("SQUARES_BEST_DIR", str(tmp_path))
    jobs = make_jobs(range(1, 9), 1, [0])
    done = []

    def runner(k):
        leases = Leases(tmp_path / "leases", ttl=30)
        leases.owner = f"runner{k}"
        done.append(run(jobs, leases, "tests.fake_solver", ["--sleep", "0.3"], 2, 30, tmp_path / f"logs{k}"))

    threads = [threading.Thread(target=runner, args=(k,)) for k in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    ran = sorted(p.name.split("_")[0] for p in (tmp_path / "ran").iterdir())
    assert ran == [str(n) for n in range(1, 9)]  # every job ran exactly once between them
    assert all(d["state"] == "done" for r in done for d in r)
    assert all(json.loads(p.read_text())["state"] == "done" for p in (tmp_path / "leases").glob("*.json"))


def test_concurrent_offers_keep_the_shortest(tmp_path):
    # The n = 5 optimum spread out by different amounts: valid, all shorter than the grid.
    x, y, deg, _ = polish(*five_solution(), gap=1e-9)
    results = []

    def offer(scale):
        results.append(best_store.offer(5, np.asarray(x) * scale, np.asarray(y) * scale, deg, f"s{scale}",
                                        root=tmp_path))

    threads = [threading.Thread(target=offer, args=(1.0 + 0.01 * k,)) for k in range(1, 9)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert best_store.load(5, tmp_path)["source"] == "s1.01"
    assert not list(tmp_path.glob("*.lock")) and not list(tmp_path.glob("*.tmp*"))
