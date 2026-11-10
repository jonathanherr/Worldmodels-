# Graded band ends vs the leaders' sides (2026-11-10)

Hypothesis tested: the leaders' L-k falls with k because band-end cost per row falls with the
band tilt (a Read-Shockley-like law for low-angle grain boundaries), so taller, shallower bands
get cheaper ends.

Method (`implied_c.py`): for every trailing n = k^2+m, invert the one-band model
f(k-r) + (k+f) sin(phi) + 2c(r+f) = 2kf + f^2 - m at the Kaggle per-n best f, taking the band
height r that needs the least good ends, and read off the implied end cost c per row per end.

Result (leaders, 414 n): c is flat. Median 0.380 (5-95%: 0.346-0.401). By tilt: 8-11 deg 0.383,
11-14 0.391, 14-17 0.382, 17-20 0.364. By k: 0.384 at k 16-19, 0.379 at k 28-31. No fall towards
small angles, so the graded/low-angle-boundary law is rejected.
Ours: median 0.402, rising with k (0.394 -> 0.410) because our f plateaus.
Implied band height is r = k-1 (whole-box band) for most n.

So the leaders' "knob" is band height growing with k, with an end that costs ~0.38/row at every
height and tilt. Our best exact ends reach that only at r = 7-8; at r >= 10 they cost 0.42-0.50.
Model value (optimistic, no integer rounding): c=0.38 everywhere -> +0.303 over our file;
c=0.39 -> +0.186.
