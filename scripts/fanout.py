"""Fan many solver runs out over the n range, on one machine or on many sharing one store.

Every copy of this script pointed at the same SQUARES_BEST_DIR works from the same job list and
splits it with lease files, so starting more copies (more processes, machines or agents) just
finishes the list sooner:

    # 8 solver processes on this machine, n = 101..1000 in chunks of 10, two seeds each
    SQUARES_BEST_DIR=/shared/best python -m scripts.fanout --n 101-1000 --chunk 10 --seeds 0,1 --procs 8

    # Same list from a second machine: it skips what the first one claimed or finished
    SQUARES_BEST_DIR=/shared/best python -m scripts.fanout --n 101-1000 --chunk 10 --seeds 0,1 --procs 8

    # Anything after the fan-out options goes to the solver (scripts.evolve_squares by default)
    python -m scripts.fanout --n 30-60 -- --per-n --from-store --gens 20

A job is one chunk of n with one seed, run as `python -m <module> --n a-b --seed s --workers 1
<solver args>`. Jobs go worst-first by the chunk's current score (sum of L / sqrt(n) - 1 from the
store), so the puzzles with the most to gain are tried first. A job is claimed by creating
`<store>/leases/<tag>/<job>.json` exclusively; the runner rewrites it every few seconds, and a
lease not rewritten for --lease-ttl seconds belongs to a runner that died and is taken over.
Finished jobs (done, failed or timed out) stay marked and are skipped unless --rerun. The tag
defaults to a hash of the module and solver arguments, so a different solver setup is a fresh
job list over the same n.

Leases need a filesystem where exclusive create is atomic (a local disk or NFS). On a folder
that syncs between machines lazily two runners can occasionally take the same job; that only
wastes time, since the store keeps the shortest certified packing whoever writes it.
"""

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

from wm import best_store
from wm.squares import puzzle_score

REPO = Path(__file__).resolve().parent.parent


def parse_ns(spec: str) -> list[int]:
    ns = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        ns.extend(range(int(a), int(b or a) + 1))
    return ns


def make_jobs(ns, chunk: int, seeds, root: Path | None = None) -> list[dict]:
    """Jobs (chunk of n x seed), the chunks with the worst current score first."""
    ns = sorted(set(ns))
    chunks = [ns[k:k + chunk] for k in range(0, len(ns), chunk)]
    gain = {tuple(c): sum(puzzle_score(best_store.best_side(n, root), n) for n in c) for c in chunks}
    chunks.sort(key=lambda c: -gain[tuple(c)])
    # Seeds outermost: every chunk gets its first seed before any chunk gets a second.
    return [{"id": f"n{c[0]:04d}-{c[-1]:04d}_s{s}", "ns": c, "seed": s, "score": gain[tuple(c)]}
            for s in seeds for c in chunks]


def spec(ns) -> str:
    return ",".join(str(n) for n in ns)


class Leases:
    """Exclusive-create lease files, one per job, under `<dir>/<job id>.json`."""

    def __init__(self, root: Path, ttl: float):
        self.dir, self.ttl = root, ttl
        self.dir.mkdir(parents=True, exist_ok=True)
        self.owner = f"{socket.gethostname()}:{os.getpid()}"

    def _path(self, job_id: str) -> Path:
        return self.dir / f"{job_id}.json"

    def read(self, job_id: str) -> dict | None:
        try:
            return json.loads(self._path(job_id).read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def _write(self, job_id: str, rec: dict):
        p = self._path(job_id)
        tmp = p.with_suffix(f".tmp{os.getpid()}-{uuid.uuid4().hex[:8]}")
        tmp.write_text(json.dumps(rec))
        os.replace(tmp, p)

    def claim(self, job: dict, rerun: bool = False) -> bool:
        """Take the job if nobody holds a live lease on it (and, without rerun, nobody finished it)."""
        p = self._path(job["id"])
        rec = self.read(job["id"])
        if rec is not None:
            finished = rec["state"] != "running"
            stale = rec["state"] == "running" and time.time() - rec["heartbeat"] > self.ttl
            if not (stale or (finished and rerun)):
                return False
            # Move the old lease aside; only one of several runners doing this at once succeeds.
            try:
                os.rename(p, p.with_suffix(f".old{os.getpid()}"))
            except FileNotFoundError:
                return False
            p.with_suffix(f".old{os.getpid()}").unlink(missing_ok=True)
        rec = {"job": job["id"], "ns": job["ns"], "seed": job["seed"], "owner": self.owner,
               "state": "running", "started": time.time(), "heartbeat": time.time()}
        try:
            fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps(rec))
        return True

    def renew(self, job_id: str) -> bool:
        """Rewrite the heartbeat; False if the lease was taken over meanwhile."""
        rec = self.read(job_id)
        if rec is None or rec["owner"] != self.owner:
            return False
        rec["heartbeat"] = time.time()
        self._write(job_id, rec)
        return True

    def finish(self, job_id: str, state: str, **info):
        rec = self.read(job_id)
        if rec is None or rec["owner"] != self.owner:
            return
        rec.update(state=state, heartbeat=time.time(), ended=time.time(), **info)
        self._write(job_id, rec)


def solver_cmd(module: str, job: dict, solver_args) -> list[str]:
    return [sys.executable, "-m", module, "--n", spec(job["ns"]), "--seed", str(job["seed"]),
            "--workers", "1", *solver_args]


def run(jobs, leases: Leases, module: str, solver_args, procs: int, timeout: float, logs: Path,
        rerun: bool = False, cwd: Path = REPO) -> list[dict]:
    """Run claimable jobs, `procs` at a time, until none is left; returns the finished records."""
    logs.mkdir(parents=True, exist_ok=True)
    pending, running, finished = list(jobs), {}, []
    last_renew = time.time()
    while pending or running:
        while pending and len(running) < procs:
            job = pending.pop(0)
            if not leases.claim(job, rerun):
                continue
            log = open(logs / f"{job['id']}.log", "w")
            proc = subprocess.Popen(solver_cmd(module, job, solver_args), cwd=cwd, stdout=log,
                                    stderr=subprocess.STDOUT, env={**os.environ, "SQUARES_BEST_DIR": str(best_store.store_dir())})
            running[job["id"]] = (proc, job, log, time.time())
            print(f"start {job['id']} (score {job['score']:.4f})", flush=True)
        time.sleep(0.2)
        for job_id, (proc, job, log, t0) in list(running.items()):
            rc = proc.poll()
            if rc is None and time.time() - t0 > timeout:
                proc.kill()
                rc = proc.wait()
                state = "timeout"
            elif rc is None:
                continue
            else:
                state = "done" if rc == 0 else "failed"
            log.close()
            del running[job_id]
            leases.finish(job_id, state, returncode=rc, seconds=round(time.time() - t0, 1))
            finished.append({"id": job_id, "state": state, "returncode": rc})
            print(f"{state:7s} {job_id} after {time.time() - t0:.0f}s", flush=True)
        if time.time() - last_renew > leases.ttl / 4:
            for job_id, (proc, *_rest) in list(running.items()):
                if not leases.renew(job_id):
                    # Someone took this job over (we looked dead to them); let them have it.
                    proc.kill()
                    print(f"lost    {job_id}", flush=True)
            last_renew = time.time()
    return finished


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--n", required=True, help="sizes, e.g. 101-1000 or 17,29,54")
    p.add_argument("--chunk", type=int, default=1, help="n per job")
    p.add_argument("--seeds", default="0", help="seeds per chunk, e.g. 0,1,2 or 0-3")
    p.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    p.add_argument("--module", default="scripts.evolve_squares", help="solver run as python -m MODULE")
    p.add_argument("--timeout", type=float, default=3600.0, help="seconds per job before it is killed")
    p.add_argument("--lease-ttl", type=float, default=120.0)
    p.add_argument("--tag", default=None, help="job-list name (default: hash of module and solver args)")
    p.add_argument("--rerun", action="store_true", help="also take jobs that already finished")
    p.add_argument("--dry-run", action="store_true", help="print the job order and lease states only")
    p.add_argument("--logs", default="runs/fanout")
    args, solver_args = p.parse_known_args()
    if solver_args[:1] == ["--"]:
        solver_args = solver_args[1:]

    tag = args.tag or hashlib.sha1(json.dumps([args.module, solver_args]).encode()).hexdigest()[:8]
    leases = Leases(best_store.store_dir() / "leases" / tag, args.lease_ttl)
    jobs = make_jobs(parse_ns(args.n), args.chunk, parse_ns(args.seeds))
    print(f"{len(jobs)} jobs, tag {tag}, store {best_store.store_dir()}, store total {best_store.total_score():.4f}")
    if args.dry_run:
        for job in jobs:
            rec = leases.read(job["id"]) or {}
            print(f"{job['id']}  score {job['score']:.4f}  {rec.get('state', '-')}")
        sys.exit(0)
    t0 = time.time()
    done = run(jobs, leases, args.module, solver_args, args.procs, args.timeout,
               Path(args.logs) / tag, args.rerun)
    states = {s: sum(d["state"] == s for d in done) for s in ("done", "failed", "timeout")}
    print(f"ran {len(done)} jobs in {time.time() - t0:.0f}s {states}; store total now {best_store.total_score():.4f}")
