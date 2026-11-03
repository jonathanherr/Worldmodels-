# 45-degree band seams (2026-11-03)

Question: can the half-cell triangles where a 45-degree band of squares meets the axis grid
(12.6% of empty area in tight n >= 100, per the negative-space atlas) be closed by a better seam?

Findings
- Every straight 45-degree band design has side K + f with f = (w*sqrt2 - j)/2 (w = band width in
  tilted rows, j = extra cell diagonals removed). Per diagonal step it gains (a+b) - 1 squares,
  where a+b is the block shift and the "-1" is exactly the two triangles. A band needs width >= 1,
  so f cannot go below the formula. 147 of our n (and 112 of the leaders') lie exactly on it:
  (1,0)=0.7071 single diagonal, (8,10)=0.6569 tilted block, (5,6), (11,14), (13,17)=0.6924, (18,24)=0.7279, ...
- The triangle-heavy n mostly already equal the Kaggle best (e.g. 493, 538, 686, 740, 853).
  Leaders beat our 45-band designs at 40 n, worth 0.0153 total, using other (non-45) designs.
- seamwaste.py: waste per unit length of a straight seam between an axis grid and a 45-degree grid
  as a function of seam direction. Seams at 0 and 45 degrees waste 0.354 (the triangles); every
  direction in between wastes 0.57-0.77. No straight seam gadget beats the triangles.
