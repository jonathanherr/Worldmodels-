# Crystal-defect / CSL angle catalog for k^2+m bands (2026-11-04)

Idea: treat k^2+m as a square crystal with m extra atoms; the tilted band is a grain, its ends a
grain boundary against the wall. Mirror-wall argument: a band leaning by phi meets a vertical wall
like a symmetric tilt boundary of misorientation 2phi, which is a coincidence-site-lattice (CSL)
angle exactly when tan(phi) = p/q. At those angles the band lattice has an exact vertical period, so
ends could interlock. Band height then fixes f(r) = ((r+1)q + p)/sqrt(p^2+q^2) - r.

Files
- `catalog.py`: 277 designs, Sigma <= 100, r 3..30, f .40-.98.
- `exact_end.py`: end DP evaluated at every contact breakpoint of the column phase a0 (exact
  coincidence phases) and 24 row offsets, giving the exact (N, face) Pareto front per end.
- `scan_catalog.py`: one band or horizontal+vertical band, every n=k^2+m, k<=31 vs our best file.

Results
- Exact contact phases give the same counts as the old 80-point grid tables (k=29, r 7-10,
  f .505-.52): grid resolution was not hiding squares.
- Catalog scan: 0 wins; no design within 1 square; 32 designs 2 squares short. Notable: tan phi = 1/2,
  r=8, f=.49706 (below leaders' .50215) is exactly 2 short at k^2+1 for every k=26..31;
  tan=1/3 r=15 f=.49516 2 short at 962.
- Best file tilts (108,802 non-axis squares): 45 deg sits on CSL (1/1); the 18-19 deg family
  scatters 18.1-18.9 around atan(1/3)=18.43, i.e. vicinal, not exact, because the band height
  pins phi. Leaders' small-m sides (.50215, .51105) are not of the CSL form A/sqrt(s) - c (s<=400).
- Conclusion: exact coincidence buys nothing in the band model; the shortfall is the ends (need
  +1 square per end at r=8, tan 1/2), same lever as research/corner_scale.
