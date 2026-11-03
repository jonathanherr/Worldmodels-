# Corner pieces that scale with box size (2026-11-03)

Goal: k^2+m packings whose extra side f = L - k shrinks with k, like the per-n leaders.

Findings
- Couzo cert n=1765 (k=42, m=1, f=0.4877) and n=2043 (k=45, m=18) are full-width bands of tilted
  columns (r+1 squares tilted phi with (r+1)cos phi + sin phi = r+f), axis rows above, sometimes a
  second vertical band.
- Waste model: f(k-r) + (k+f) sin phi + E_ends <= 2kf + f^2 - m. The leaders' f at every k=20..31,
  m=1..25 matches end waste ~0.39 per row per end. Couzo's ends ~0.44.
- `dpend.py`/`fastdp.py`: exact staircase DP for an end (axis rows + truncated columns). Best
  0.43/row at r=9; some (r,f) are luckier (0.35-0.39) but carry large sin phi.
- `cut_band.py`: shrinking n1765 to k=29 is 13 squares short.
- `endtable.py` + `scan2.py`: tabulate ends for r=4..24, f grid, and scan every n=k^2+m for one- or
  two-band designs that beat our side.
