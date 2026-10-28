"""Program search CLI: show a task, score a program, post it to the board, list the leaders, brief an agent.

  python -m scripts.progsearch ctx corner147 --n 147        # the window as a program sees it
  python -m scripts.progsearch eval prog.py corner147         # score without posting
  python -m scripts.progsearch submit prog.py corner147 --author jammer --note "..." [--parent ID]
  python -m scripts.progsearch top corner147
  python -m scripts.progsearch prompt corner147 --persona escher   # brief for a Claude session

Board: $PROGSEARCH_BOARD (default /mnt/project-files/squares/progsearch). Seeds: $PROGSEARCH_SEEDS.
Wins are offered to the store in $SQUARES_BEST_DIR when --offer is given.
"""

import argparse
import json
import os
from pathlib import Path

from wm import progsearch as ps

# Design personas only: the 2026-10-27 audit found jammer (loosen and compress = seeded shake) and
# darwin (small mutations of the best program) repeat dead ends when record basins snap back.
PERSONAS = {
    "erdos": "Think in exact constructions: tilted blocks at angles with clean trigonometry, rows that tile a wedge.",
    "escher": "Think in tilings and symmetry: a periodic cell that repeats along the band, then design its two ends.",
    "lagrange": "Think in contacts: write down which contacts bind the side (or f), solve that rigid system exactly, "
                "then change the contact topology so a different system binds lower.",
    "architect": "Think in whole layouts: two strips, two angles, a band that meets the walls at a different angle, "
                 "staircases with uneven steps. Sketch the layout on paper first, then compute it.",
}

# Approaches already run on this problem that found nothing (or only crumbs). Every brief carries it.
DONT_BOTHER = """## Don't bother (already tried, no wins)
- Perturbing, shaking, polishing or relaxing a record packing: record basins snap back. Seeded shake won 0 of 12 n,
  GPU perturb+relax on all records gained 0.003 points in total, program-search window edits scored 0.
- Small edits of the strip records (299/335/412): the strip angle and g = (sqrt7 - 1)/2 are optimal for that design;
  emptying a 4 x 4 corner window frees no side at all.
- Turning the 147 diamond off 45 degrees: it always snaps back.
- Rubik slice/block turns (0 of 2176), soft-shape wiggling, drunk-walk grid starts, one agent per square, Game-of-Life
  local rules, rectangle decomposition, 3D lift, neuroevolution from the grid, shelves of smaller-n record blocks:
  all 0 wins against the records. Local rules do not assemble the global designs the records use.
- Grow-and-cut sticks at band ends: the slack is fragmented (at most 0.18 of a square each).
- Gadget compaction of existing gadgets (lowers f by ~1e-4), looser tip rules, longer chains: no change.
- Anything that starts from the record's own layout. The construct tasks hide it on purpose.
"""

FACTS = {
    "construct-strip": """## Established facts
- Record: a two-wide strip at 24.295 deg between two axis-aligned staircases; side = k + 0.822876.
- Kaggle bests are 0.020-0.025 smaller (299: 17.802719, 412: 20.798166), and the same f ~ 0.796-0.799 appears at
  454, 498, 544, 592, 641, 693, 747, 803, 861, 982 (all n = K^2 + about K - 8). So one band design, better than
  the record's, works for every K: it is a periodic design, not a one-off.
- Lead from the last run: a periodic band cell of 16 squares per 6 staircase steps (dominoes at mixed 26/30 deg)
  wastes 0.981 per step vs the record's 1.000, but naive ends gave back the gain (f 0.82365). Design the ends.
""",
    "construct335": """## Established facts
- Record side 18.822876 (the 299/412 strip design); another team has 18.794904.
- 335 = 18^2 + 11 needs about 2.7 extra squares per unit of staircase, more than the strip family carries;
  the last run's agents died on 335 when it was bundled with 299/412.
- About 6.7 of the record's 19.3 units of empty area sit in the two 4 x 4 strip ends.
""",
    "construct147": """## Established facts
- Record 12.656854 = 148's layout (45-degree 8 x 8 diamond in an axis frame) minus one corner square. Kaggle best 12.651044.
- The frame staircase corners touch the diamond on all four sides, so the gain needs a different design, probably
  two tilt angles (e.g. a diamond that is not square, or a second tilted group near the free corner).
""",
    "gadget": """## Established facts
- Kaggle per-puzzle bests sit at side K + 0.795-0.799 for n = K^2 + D with D about K - 8 (299, 412, 454, 498, 544,
  592, 641, 693, 747, 803, 861, 982); our best chains reach only K + 0.823-0.828 there. That is the largest single
  family in the 0.149-point gap to the per-puzzle bests (318 n trail them).
- The records' gadgets are already in the pool, so a gadget only scores if its f beats the pool's best f for some
  (w, h, d) that a chain needs. `python -m scripts.gadget_search table` shows which.
""",
}

def brief(task, persona=None, top=3):
    spec = ps.TASKS[task]
    if spec.get("kind") == "gadget":
        return gadget_brief(task, persona, top)
    lines = [f"# Program search task: {task}", "", spec["goal"], "",
             f"n values: {spec['ns']}. " + ("Window: every tilted square; coordinates are the box's own." if spec.get("select") == "tilted" else f"Window: {spec['size']} x {spec['size']} at the {spec['corner']} corner."),
             "Write a Python file defining build(ctx) that returns exactly ctx['k'] squares (x, y, angle_deg)",
             "in window-local coordinates: box walls are x = 0 and y = 0 (and x = y = side for whole-block tasks), corner windows are [0, size]^2.",
             "ctx keys: n, k, size, side (record side), target_side (best side another team reached),",
             "original (record's window squares; empty in construct tasks), frozen (fixed neighbours you must not overlap).",
             "Helpers: from wm.progsearch_lib import block, rotate, translate, worst_overlap, settle, waste_in.",
             "Your squares replace the window, the whole packing is squeezed from the record side, and the score",
             "is record side - final side summed over the n values (0 = no change, > 0 = a win, -1 = broken).",
             "Limits: 60 s CPU per n, 2 GB memory, no network, no files. Keep programs under 150 lines.",
             "Run: python -m scripts.progsearch ctx TASK --n N, then eval PROG TASK, then submit PROG TASK --author YOU.",
             ""]
    if persona:
        lines += [f"Your angle: {PERSONAS[persona]}", ""]
    lines += [FACTS.get(task, ""), DONT_BOTHER]
    lines += _leaders(task, top)
    return "\n".join(lines)


def gadget_brief(task, persona=None, top=3):
    lines = [f"# Program search task: {task}", "", ps.TASKS[task]["goal"], "",
             "Write a Python file that sets KEY = (w, h, d) and defines build(ctx) returning exactly w*h + d squares",
             "(x, y, angle_deg) inside [0, w + f] x [0, h + f]. ctx carries nothing you need. The harness squeezes the",
             "squares to the smallest feasible f, adds every corner crop to the gadget pool of our submission and reruns",
             "the fault-line DP (~20 s). Score = points the whole submission gains, checked with the official metric",
             "(0 = no n improved). The result also shows your f and the best f the pool already has for that KEY.",
             "Helpers: wm.progsearch_lib (block, rotate, translate, worst_overlap, settle), wm.gadget_opt (squeeze, improve),",
             "wm.fault_dp (extract, symmetries). Limits: 60 s CPU, 2 GB memory, no network, no files.",
             "Run: python -m scripts.gadget_search table, then python -m scripts.progsearch eval PROG gadget, then submit.",
             ""]
    if persona:
        lines += [f"Your angle: {PERSONAS[persona]}", ""]
    lines += [FACTS["gadget"], DONT_BOTHER]
    lines += _leaders(task, top)
    return "\n".join(lines)


def _leaders(task, top):
    lines = []
    rows = ps.leaderboard(task, top=top)
    if rows:
        lines.append("## Current leaders")
        for r in rows:
            code = (ps.BOARD / task / f"{r['id']}.py").read_text()
            lines += [f"### {r['id']} score {r['score']:+.6f} by {r['author']}: {r['note']}", "```python", code.rstrip(), "```"]
    return lines


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("ctx"); c.add_argument("task"); c.add_argument("--n", type=int)
    e = sub.add_parser("eval"); e.add_argument("prog"); e.add_argument("task"); e.add_argument("--offer", action="store_true")
    s = sub.add_parser("submit"); s.add_argument("prog"); s.add_argument("task"); s.add_argument("--author", required=True)
    s.add_argument("--note", default=""); s.add_argument("--parent"); s.add_argument("--offer", action="store_true")
    t = sub.add_parser("top"); t.add_argument("task"); t.add_argument("--k", type=int, default=10)
    h = sub.add_parser("headroom"); h.add_argument("task")
    p = sub.add_parser("prompt"); p.add_argument("task"); p.add_argument("--persona", choices=sorted(PERSONAS))
    a = ap.parse_args()
    offer_root = Path(os.environ["SQUARES_BEST_DIR"]) if getattr(a, "offer", False) and "SQUARES_BEST_DIR" in os.environ else None
    if a.cmd == "ctx":
        spec = ps.TASKS[a.task]
        if spec.get("kind") == "gadget":
            print(spec["goal"])
            return
        ctx = ps.make_ctx(a.n or spec["ns"][0], spec["corner"], spec["size"], goal=spec["goal"], select=spec.get("select", "window"),
                          hide=spec.get("hide", False))
        print(json.dumps({k: v for k, v in ctx.items() if not k.startswith("_")}, indent=1))
    elif a.cmd == "eval":
        print(json.dumps(ps.evaluate(Path(a.prog), a.task, offer_root=offer_root), indent=1, default=str))
    elif a.cmd == "submit":
        r = ps.submit(Path(a.prog), a.task, a.author, a.note, a.parent, offer_root=offer_root)
        print(json.dumps(r, indent=1, default=str))
    elif a.cmd == "top":
        for r in ps.leaderboard(a.task, top=a.k):
            print(f"{r['id']}  {r['score']:+.6f}  {r['author']:10s}  {r['note']}")
    elif a.cmd == "headroom":
        print(json.dumps(ps.headroom(a.task), indent=1))
    elif a.cmd == "prompt":
        print(brief(a.task, a.persona))


if __name__ == "__main__":
    main()
