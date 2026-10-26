"""Baseline for construct-* tasks: return the record packing itself (scores about 0)."""


def build(ctx):
    return [tuple(s) for s in ctx["original"]]
