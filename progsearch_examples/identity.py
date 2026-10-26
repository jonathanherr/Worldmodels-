"""Baseline: return the record's own window layout (scores about 0)."""


def build(ctx):
    return [tuple(s) for s in ctx["original"]]
