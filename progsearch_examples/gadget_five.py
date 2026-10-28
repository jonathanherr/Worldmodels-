"""Gadget example: five squares in a 2 x 2 corner rectangle (the n = 5 packing, f = sqrt(2)/2).

The pool already has this gadget, so it scores 0; it shows the file shape for the "gadget" task.
"""

import math

KEY = (2, 2, 1)  # w, h, d: w*h + d squares in [0, w + f] x [0, h + f]


def build(ctx):
    a = 1.5 + math.sqrt(2) / 2  # centre of the far corner squares at side 2 + sqrt(2)/2
    c = (2 + math.sqrt(2) / 2) / 2
    return [(0.5, 0.5, 0.0), (a, 0.5, 0.0), (0.5, a, 0.0), (a, a, 0.0), (c, c, 45.0)]
