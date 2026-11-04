# Torus-then-cut probe (2026-11-04)

Idea: pack n unit squares on a torus (no walls), cut it open into an L x L box, repair the seam.

* Wall-free (torus) packings are trivial: rows of squares at angle t with free per-row sliding
  tile a torus of side sqrt(a^2+b^2) perfectly. So the torus step carries no information;
  every bit of waste is created by the cut.
* `torus_window.py`: window a tilted sliding-row lattice with a box of side L (t 0.25-45 deg,
  row phase, per-row best slide), L = 2..33 step 0.005. `runrepair.py`/`repair.py`: greedily add
  axis-aligned squares in the cut border (the seam repair), 12 best windows x 3 slide alignments.
* Result vs submission_merged_fault_20261028 (22.337944): 0 wins at any n. Raw window beats plain
  grid at some n but trails ours everywhere; at the record side the window is >= 1 square short at
  all 546 non-grid n (median 5). Repair added a square in 4 of 118 runs and never closed the gap.
* One-seam (cylinder) version: cutting a periodic band once leaves a left and a right wall, i.e.
  exactly the two band ends of the leaders' design. Periodicity only forces the two ends to be
  complementary pieces of one lattice, a restriction, so it cannot beat the free band-end search
  (research/corner_scale, 0 wins).
