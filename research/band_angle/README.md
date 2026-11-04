# Band tilt angle as a free variable (2026-11-04)

Question: in the one-band k^2+m design (research/corner_scale), the tilt is fixed to the tight angle
phi*(r, f) with (r+1) cos phi + sin phi = r + f. Does freeing the angle line up the two band ends so one
more square fits (k^2+1 at k 20-31, 577 and 962 were 1 square short)?

`anglescan.py`: exact staircase DP ends at any phi >= phi* (below phi* a full column no longer fits),
200 a0 x 8 offsets x 2 kinds per end, both ends' phases chosen independently. Scan r 5-16,
f .48-.56 step .004, phi* + 0..0.06 rad in 31 steps (7,812 cells, ~12 min on 4 CPUs).

    python -m research.band_angle.anglescan OUT.jsonl 5 16 0.48 0.56 0.004 0.06 31
    python -m research.band_angle.analyze OUT.jsonl

Result: 0 n where a free angle lowers the reachable f, 0 wins.
- Over all (k, f) cells the best count with free angle equals the tight-angle count at 351 of 357; at 6
  cells (k 16-18, f .54-.544, r=8, +0.004-0.006 rad) it gains 1 square, none of them binding.
- Why: D = L cos phi - H sin phi (columns that fit between the walls) is largest at phi*, and band interior
  waste H - (r+1) cos phi grows above sin phi. Steeper tilt only reshuffles the end staircases, which the
  free a0 phase already does.
- Per-square angle freedom at the ends was already tested by the CPU squeeze (endgadget.Gadget frees t),
  which moved the face by <= 0.011.
- k^2+1 stays 1 square short at our f (.5208 for k >= 25, .5355 below). The leaders' f ~.502 needs a
  different structure, not a re-tilted band.
