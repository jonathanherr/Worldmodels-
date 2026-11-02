# Formula fits of finished packings (2026-11-02)

Question: can the finished (best-known) packings be written as formulas in n (or k, offset d)
and those formulas used to solve other n? Run from a scratch dir holding `best.pkl` (written by
`load.py` from the best submission CSV) and `sides.json`.

| script | idea | result |
| --- | --- | --- |
| `ident.py` | fit low-degree integer polynomials / sqrt-combos to the Kaggle per-puzzle best sides | only ~10 digits known; every hit is spurious (no clean algebraic form) |
| `fam.py` | treat side fraction f(K, D) as linear along lines in (K, D); interpolate / extrapolate | 76 interpolation targets, all coincidences of equal f from different designs; extrapolation predicts values below proven optima |
| `shrink.py`, `shrink2.py` | remove a clean row band + column band (straight or staircase) -> side-1 | 446 packings shrinkable, 0 improvements: our file is closed under it (and under L-wrap) |
| `blockfit.py`, `blockmis.py`, `blocksearch.py` | parametric single block a x b at any angle + 4 corner-anchored axis lattices, MILP max fill | reproduces n=366 (15x8 at 45 deg); at L 0.00085 below ours the best over 41-49 deg is 361 squares, so rigid off-45 blocks never help |

The leaders beat our 45-degree block family (f = 4 sqrt2 - 5 = 0.65685) at 147, 329, 366, 447,
490, 535, 583, 632, 683 with f -> 0.645802 as k grows; that design needs non-lattice axis cells
or a deformed block, not a rotated rigid one.
