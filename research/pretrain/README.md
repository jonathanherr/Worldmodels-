# Pretraining data: band-end cropper (2026-11-03)

Depends on research/corner_scale, copied here from branch claude/project-thread-mlnmr2 at b78c556
(that branch stays the source of truth for it).

`end_crop.py` finds full-width tilted bands (one tilt cluster spanning > 0.8 L, < 0.6 L tall) in
packings and measures each end's waste per row, c_end = (half-slab waste - (L/2) sin phi) / H,
the same per-row-per-end unit as research/corner_scale. It also crops each end into band
coordinates (wall at u=0, band bottom at v=0, right ends mirrored) for a training corpus.

    python research/pretrain/end_crop.py --csv BEST.csv --certs COUZO_CERT_DIR --out OUT

First run (our best file 22.337944 + 332 Couzo certs with n > 1000):
- 346 tilt clusters, only 8 wall-to-wall bands (6 ours, Couzo n1765 and n2043), 16 ends.
- Ends below 0.40 per row exist (ours 978 hi 0.239, 503 lo 0.324, 749 lo 0.326, Couzo 2043 lo
  0.379), but only at f 0.70-0.96 and steep tilts, where the band interior (L sin phi) costs more.
  At small m (f ~ 0.5) the best real end is Couzo 1765 at 0.43.
- Band-model screen with the best end used at both ends, k 12-31: 0 n beat the leaders, 0 beat ours.
- So c alone is the wrong label: an end's value depends on (phi, f, r) jointly. Label ends by the
  m they allow at each k, not by c.

## Synthetic ends, critic and GA (2026-11-03, cloud CPU)

- `synth_ends.py dp`: exact DP ends for r 5-20, f 0.45-1.00 step 0.005, 80 a0 x 8 offsets x 2 kinds
  = 2.27M ends (dp_r*.npz, ~2 min on 4 cores). `rescan.py`: one-band designs for every k^2+m,
  k <= 31: 0 wins; closest are 1-2 squares short (k^2+1 at k 20-31 with r=8, f .52-.535).
- `synth_ends.py squeeze`: CPU Gadget.squeeze of the best DP ends moves the face by <= 0.011
  (66 ends), far from the ~1 needed for an extra column. Set OMP/OPENBLAS/NUMBA threads to 1 or
  each squeeze takes 330 s instead of 5 s.
- `ga_ends.py`: insert/relocate moves on near-miss ends, online gradient-boosting critic, scored
  by band count. 78 min, 2,102 exact evals, 56 improved ends (27 with an extra square), best
  0.3815/row at r=8 f=.535. Shortfalls: n=962 and n=577 went from 2 to 1 square short; 0 wins.
- Critic: held-out rank correlation 0.59 on waste change (mostly feasibility), but its picks
  gained less often than random (2.6% vs 4.5%). Hole features don't predict which inserts pay.
