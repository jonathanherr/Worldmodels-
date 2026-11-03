# Pretraining data: band-end cropper (2026-11-03)

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
