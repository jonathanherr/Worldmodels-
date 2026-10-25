"""Run the packing research swarm, or act on its board by hand.

    # 7 personas, one process each, 60 minutes, 90 s per run, n <= 300
    SQUARES_BEST_DIR=/shared/best SWARM_BOARD=/shared/swarm \
        python -m scripts.swarm run --agents erdos,jammer,darwin,lagrange,metropolis,escher,bandit --minutes 60

    # several agents per persona: name them persona or persona:count
    python -m scripts.swarm run --agents metropolis:3,darwin:2,bandit:2 --minutes 120

    python -m scripts.swarm digest                 # print (and write) the board summary
    python -m scripts.swarm tools                  # the tool table
    python -m scripts.swarm personas               # the roster
    python -m scripts.swarm prompt jammer          # instructions for a Claude session playing jammer
    python -m scripts.swarm try --agent me --n 37 --recipe '[{"t":"store"},{"t":"jolt"},{"t":"squeeze_box"}]'
    python -m scripts.swarm note --agent jonathan --text "try lattice with theta near 40 at n=55"

Every agent points at the same store (SQUARES_BEST_DIR) and board (SWARM_BOARD), so agents on
several machines or Claude sessions acting by hand all share results.
"""

import argparse
import json
import multiprocessing as mp
import time

from wm.swarm.agents import PERSONAS, describe_personas, persona_prompt, run_agent, summarize
from wm.swarm.board import Board
from wm.swarm.tools import describe, evaluate, recipe_id


def parse_agents(spec: str):
    out = []
    for part in spec.split(","):
        name, _, count = part.partition(":")
        if name not in PERSONAS:
            raise SystemExit(f"unknown persona {name!r}; have {', '.join(PERSONAS)}")
        k = int(count or 1)
        out += [(f"{name}{i if k > 1 else ''}", name) for i in range(k)]
    return out


def cmd_run(a):
    board = Board(a.board)
    agents = parse_agents(a.agents)
    board.post("swarm", "note", f"swarm started: {', '.join(n for n, _ in agents)} for {a.minutes} min, "
               f"budget {a.budget}s, n <= {a.max_n}")
    jobs = [(name, persona, str(board.root), a.seed * 1000 + i, a.minutes, a.budget, a.max_n)
            for i, (name, persona) in enumerate(agents)]
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=run_agent, args=(j,), daemon=False) for j in jobs]
    for p in procs:
        p.start()
    t_digest = 0.0
    while any(p.is_alive() for p in procs):
        if time.time() - t_digest > a.digest_every:
            board.write_digest()
            t_digest = time.time()
        time.sleep(5)
    print(board.digest())
    board.write_digest()


def cmd_try(a):
    board = Board(a.board)
    recipe = json.loads(a.recipe)
    rid = recipe_id(recipe)
    if rid not in board.recipes():
        board.post(a.agent, "recipe", summarize(recipe), a.persona, recipe=recipe, recipe_id=rid,
                   parents=a.parents.split(",") if a.parents else [], why=a.why)
    if a.hypothesis:
        lo, _, hi = a.band.partition("-")
        board.post(a.agent, "hypothesis", a.hypothesis, a.persona, recipe_id=rid, band=[int(lo), int(hi or lo)])
    board.claim(a.agent, a.n, a.budget / 60 * 1.5, a.persona)
    res = evaluate(recipe, a.n, seed=a.seed, budget=a.budget, source=f"swarm:{a.agent}:{rid}")
    board.post(a.agent, "result", f"n={a.n} {res['before']:.6f} -> {res['side']}", a.persona, **res)
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--board", default=None, help="board directory (default $SWARM_BOARD or swarm_board/)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--agents", default=",".join(PERSONAS))
    r.add_argument("--minutes", type=float, default=30)
    r.add_argument("--budget", type=float, default=90, help="seconds per recipe run")
    r.add_argument("--max-n", type=int, default=300)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--digest-every", type=float, default=60)
    t = sub.add_parser("try")
    t.add_argument("--agent", required=True)
    t.add_argument("--persona", default="human")
    t.add_argument("--n", type=int, required=True)
    t.add_argument("--recipe", required=True)
    t.add_argument("--why", default="")
    t.add_argument("--parents", default="")
    t.add_argument("--hypothesis", default="")
    t.add_argument("--band", default="")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--budget", type=float, default=120)
    n = sub.add_parser("note")
    n.add_argument("--agent", required=True)
    n.add_argument("--text", required=True)
    sub.add_parser("digest")
    sub.add_parser("tools")
    sub.add_parser("personas")
    p = sub.add_parser("prompt")
    p.add_argument("persona", choices=list(PERSONAS))
    a = ap.parse_args()
    if a.cmd == "run":
        cmd_run(a)
    elif a.cmd == "try":
        cmd_try(a)
    elif a.cmd == "note":
        Board(a.board).post(a.agent, "note", a.text)
    elif a.cmd == "digest":
        b = Board(a.board)
        print(b.digest())
        b.write_digest()
    elif a.cmd == "tools":
        print(describe())
    elif a.cmd == "personas":
        print(describe_personas())
    elif a.cmd == "prompt":
        print(persona_prompt(a.persona))


if __name__ == "__main__":
    main()
