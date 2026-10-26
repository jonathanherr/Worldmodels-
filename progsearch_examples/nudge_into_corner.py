"""Example mutation: push every window square 0.05 toward the corner, then settle against the frozen ones."""
from wm.progsearch_lib import settle


def build(ctx):
    moved = [(x - 0.05, y - 0.05, d) for x, y, d in ctx["original"]]
    return settle(moved, ctx["frozen"])
