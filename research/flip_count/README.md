# Flipped problem: how many squares fit a box of side L (2026-11-04)

Baseline: submission_merged_fault_20261028.zip (22.337944 official).

1. Monotone envelope: no n+1 packing is smaller than the n packing anywhere (0 violations), so
   "drop squares from a bigger n" gives nothing new.
2. Count view vs the Kaggle per-n leaderboard (2026-11-03): our file has 549 distinct box sizes.
   In 202 of them the leaders fit more squares than we do: 113 boxes by 1, 86 by 2, 3 by 3
   (926->929, 991->994, 992->995). Nowhere are we more than 3 squares short.
3. Fixed-L insertion (insert_fixed_L.py): box fixed just under our s(n+1), start from our n
   packing scaled up, put one square into each of the largest holes (distance transform), FIRE
   relax at that L. Run on all 202 target n+1 (6 holes x 2 angles), rerun on the 4 closest with
   20 holes x 4 angles x 2 box sizes: 0 wins. Closest residual overlap energy 1.2e-5 (n=691, box
   gap 0.0002) and 1.5e-5 (595); median 0.006. Records are jammed: an extra square needs a
   global redesign, not local relaxation (same as morph insert, 2026-10-25).
