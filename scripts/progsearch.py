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

PERSONAS = {
    "erdos": "Think in exact constructions: tilted blocks at angles with clean trigonometry, rows that tile a wedge.",
    "jammer": "Think like a granular physicist: loosen the region, compress it, and let squares find a jammed state.",
    "escher": "Think in tilings and symmetry: reflect or rotate known good sub-blocks into the window.",
    "lagrange": "Think in constraints: find which contacts bind the side and design a layout that frees them.",
    "darwin": "Mutate the current best program a little: change one angle, one count, one offset at a time.",
}


def brief(task, persona=None, top=3):
    spec = ps.TASKS[task]
    lines = [f"# Program search task: {task}", "", spec["goal"], "",
             f"n values: {spec['ns']}. Window: {spec['size']} x {spec['size']} at the {spec['corner']} corner.",
             "Write a Python file defining build(ctx) that returns exactly ctx['k'] squares (x, y, angle_deg)",
             "in window-local coordinates: box walls are x = 0 and y = 0, the window is [0, size]^2.",
             "ctx keys: n, k, size, side (record side), target_side (best side another team reached),",
             "original (record's window squares), frozen (fixed neighbours you must not overlap).",
             "Helpers: from wm.progsearch_lib import block, rotate, translate, worst_overlap, settle, waste_in.",
             "Your squares replace the window, the whole packing is squeezed from the record side, and the score",
             "is record side - final side summed over the n values (0 = no change, > 0 = a win, -1 = broken).",
             "Limits: 60 s CPU per n, 2 GB memory, no network, no files. Keep programs under 150 lines.",
             "Run: python -m scripts.progsearch ctx TASK --n N, then eval PROG TASK, then submit PROG TASK --author YOU.",
             ""]
    if persona:
        lines += [f"Your angle: {PERSONAS[persona]}", ""]
    rows = ps.leaderboard(task, top=top)
    if rows:
        lines.append("## Current leaders")
        for r in rows:
            code = (ps.BOARD / task / f"{r['id']}.py").read_text()
            lines += [f"### {r['id']} score {r['score']:+.6f} by {r['author']}: {r['note']}", "```python", code.rstrip(), "```"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("ctx"); c.add_argument("task"); c.add_argument("--n", type=int)
    e = sub.add_parser("eval"); e.add_argument("prog"); e.add_argument("task"); e.add_argument("--offer", action="store_true")
    s = sub.add_parser("submit"); s.add_argument("prog"); s.add_argument("task"); s.add_argument("--author", required=True)
    s.add_argument("--note", default=""); s.add_argument("--parent"); s.add_argument("--offer", action="store_true")
    t = sub.add_parser("top"); t.add_argument("task"); t.add_argument("--k", type=int, default=10)
    p = sub.add_parser("prompt"); p.add_argument("task"); p.add_argument("--persona", choices=sorted(PERSONAS))
    a = ap.parse_args()
    offer_root = Path(os.environ["SQUARES_BEST_DIR"]) if getattr(a, "offer", False) and "SQUARES_BEST_DIR" in os.environ else None
    if a.cmd == "ctx":
        spec = ps.TASKS[a.task]
        ctx = ps.make_ctx(a.n or spec["ns"][0], spec["corner"], spec["size"], goal=spec["goal"])
        print(json.dumps({k: v for k, v in ctx.items() if not k.startswith("_")}, indent=1))
    elif a.cmd == "eval":
        print(json.dumps(ps.evaluate(Path(a.prog), a.task, offer_root=offer_root), indent=1, default=str))
    elif a.cmd == "submit":
        r = ps.submit(Path(a.prog), a.task, a.author, a.note, a.parent, offer_root=offer_root)
        print(json.dumps(r, indent=1, default=str))
    elif a.cmd == "top":
        for r in ps.leaderboard(a.task, top=a.k):
            print(f"{r['id']}  {r['score']:+.6f}  {r['author']:10s}  {r['note']}")
    elif a.cmd == "prompt":
        print(brief(a.task, a.persona))


if __name__ == "__main__":
    main()
