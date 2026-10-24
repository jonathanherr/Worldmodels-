# World model

A small world model built from scratch, meant to grow. The plan follows the
[primer](https://claude.ai/artifact/VeLm7KtmCHPcFGeJCX8J73): start with a mini-Dreamer on a
2D environment we control, then scale up.

## What is here (v0)

| Path | What it is |
| --- | --- |
| `wm/env.py` | `BouncingBallEnv`: 64x64 ball + paddle game, 3 actions, +1 hit / -1 miss, exposes ground-truth state |
| `wm/data.py` | Random (sticky) policy data collection, `.npz` storage, sequence sampler |
| `wm/model.py` | v0 world model: conv encoder/decoder + deterministic GRU latent dynamics + reward/continue heads |
| `scripts/collect_data.py` | Collects 100k train / 10k test frames |
| `scripts/train_v0.py` | Trains v0 (CUDA, Apple MPS, or CPU) |
| `scripts/eval_v0.py` | Rollout error vs horizon, real-vs-imagined GIF, reward-head accuracy, latent probes |

## Quick start

```bash
pip install -r requirements.txt
python -m scripts.collect_data                 # ~30 s
python -m scripts.train_v0 --ch 16 --batch 16 --steps 6000 --lr 5e-4   # ~80 min on 4 CPU cores
python -m scripts.eval_v0                      # writes runs/v0/{rollout_error.png,rollouts.gif,metrics.json}
```

On a GPU use the defaults (`--ch 32 --batch 32`) and more steps.

## How v0 works

```
z_t    = encoder(x_t)
h_t+1  = GRU(h_t, [z_t, onehot(a_t)])
z^_t+1 = prior(h_t+1)            -> decoder -> predicted frame
r^, c^ = heads(h_t+1)            -> reward, episode continues
```

Training uses 32-step windows: the first 8 frames are fed in (context), and for the remaining
24 the model feeds its own predictions back in (imagination). Losses: frame reconstruction
(Bernoulli, ball and paddle pixels upweighted 20x so the small ball is not ignored), predicted-frame
reconstruction, latent matching to a stop-gradient encoder target, reward and continue heads.

## v0 results (6000 steps, 4 CPU cores, 0.7M params)

Held-out test episodes, 8 real context frames, then up to 40 imagined steps.

| Steps imagined | Pixel MSE (model) | Pixel MSE (copy last frame) | Median ball error, px (model) | Ball error if it stayed put, px | Ball missing from prediction |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.0018 | 0.0042 | 0.6 | 1.9 | 0% |
| 5 | 0.0033 | 0.0105 | 1.0 | 9.5 | 0% |
| 10 | 0.0056 | 0.0112 | 1.7 | 17.4 | 0% |
| 20 | 0.0108 | 0.0117 | 3.2 | 28.8 | 19% |
| 40 | 0.0225 | 0.0120 | 8.8* | 35.1 | 98% |

\* measured on the 2% of rollouts where a ball is still visible.

- The model tracks the ball well for ~15 steps, then the deterministic latent hedges between
  plausible futures and the ball smears into a fading blob (see `runs/v0/rollouts.gif`).
  That is the expected failure of a deterministic model and the main motivation for v1's
  stochastic latent.
- Linear probes: a single frame's latent `z` encodes positions (R^2 0.96-0.99) but not
  velocity (R^2 ~0); the GRU memory `h` encodes velocity too (R^2 0.79-0.81).
- Reward head: catches 99% of paddle hits (precision 38%); it never predicts misses, and the
  continue head never predicts episode ends. Both need work in v1.

Two training bugs found on the way, both instructive:
1. With an unweighted reconstruction loss the autoencoder never learned to draw the ball
   (plateau at "black screen"). Upweighting foreground pixels fixed it.
2. A `tanh` on the latent saturated and collapsed it to a constant; layer norm keeps the
   scale fixed without saturating.

## Squares in Squares world

A simulator for the Kaggle [Squares in Squares](https://www.kaggle.com/competitions/squares-in-squares)
problem: pack n unit squares (n = 1..1000) into the smallest axis-aligned square. A puzzle scores
`L / sqrt(n) - 1`, where L is the enclosing square's side, computed from the submitted squares
(lower is better). The leaderboard sums all 1000 puzzles; a missing or unparsable puzzle scores
as the grid packing.

| Path | What it is |
| --- | --- |
| `wm/squares.py` | Geometry and checker (separating-axis overlaps, enclosing side, score, `polish`), soft-body relaxation with contact torque, and `SquarePackingEnv` (64x64 frames + ground-truth state, 10 actions: select, move, rotate, squeeze, shake) |
| `scripts/collect_squares.py` | World-model dataset in the same layout as `wm/data.collect` |
| `scripts/solve_squares.py` | Baseline: grid vs. a squeeze-and-shake heuristic in the simulator, writes `runs/squares/{results.json,submission.csv}` |
| `web/squares.html` | Interactive bench: drag and rotate squares, auto-squeeze the box, copy the best packing as CSV |
| `tests/test_squares.py` | `python -m pytest tests` |

```bash
python -m scripts.solve_squares --n-max 12      # ~2.5 min on CPU
python -m scripts.collect_squares --n 5          # 100k train / 10k test frames, ~8 min
```

How the world works: the agent squeezes the container wall; each step runs a few relaxation passes
that push overlapping squares apart along their separating axis. Pushes act at the penetrating
corners, so off-center contacts also turn the squares, and tilted packings can appear on their
own. Reward is the drop in the best valid score seen in the episode. Frames show overlaps at full
brightness, so a world model can learn to see when a packing is invalid.

Baseline (4 seeds x 800 steps per n): it matches the grid everywhere and beats it at n = 5
(2.961 vs 3; best known 2.707) and n = 10 (3.843 vs 4; best known 3.707). Total score for
n = 1..12 is 1.889.

Submissions follow the competition's `sample_submission.csv`: `id,n,square_id,x,y,angle`, with
square centers and angles in radians, one block per n = 1..1000 (500,500 rows). The simulator
works in degrees internally and converts on export; `solve_squares.py` fills every n it did not
solve with the grid, so its `submission.csv` is always complete.
The overlap checker is ours; the official tolerance lives in Kaggle's metric notebook
(kaggle.com/code/metric/squares-in-squares), which we have not matched yet.

## Next steps

1. **v1: RSSM.** Split the latent into deterministic `h` and stochastic `z` with a KL between
   posterior and prior (DreamerV3 style), so the model can represent uncertainty.
2. **Actor-critic in imagination**, then compare its score against the random policy.
3. Harder environments: more balls, occlusion, randomness; then MiniGrid / Crafter / Atari.
