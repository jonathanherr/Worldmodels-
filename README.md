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
| `wm/kaggle_metric.py` | Kaggle's official metric, copied verbatim from the competition's metric notebook |
| `scripts/score_submission.py` | Scores a full submission CSV locally with the official metric (~3 min on 4 cores) |
| `wm/constructions.py` | Known constructions from Friedman's survey: a tilted-block family (Goebel's 45-degree diagonal strips and central blocks, off-center blocks) plus `l_wrap` |
| `wm/best_store.py` | Shared best-per-n store: one JSON per n with exact decimal strings; `offer` keeps a packing only if the official metric accepts it and it is shorter |
| `scripts/build_constructions.py` | Sweeps the construction family over sides up to 32, wraps and trims to every n, polishes, and offers each packing to the store (~25 min on 4 cores) |
| `scripts/write_submission.py` | Full 1..1000 submission CSV from the store |
| `tests/test_squares.py`, `tests/test_constructions.py` | `python -m pytest tests` |

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
The official metric certifies non-overlap with interval arithmetic, so squares that touch
exactly (like the textbook n = 5 optimum) are rejected as undecidable unless they are axis-aligned
with exact decimals. `polish` therefore leaves a 1e-9 gap between every pair, and
`solve_squares.py` scores each solved puzzle with the official metric before writing the file.
Locally the n = 1..12 baseline file scores 28.8707 with the official metric.

Known constructions (`scripts/build_constructions.py`): every packing is a block of 45-degree
squares (p rows along the main diagonal, q per row, optionally shifted) with axis-aligned staircases
grown from the four corners, which reproduces Goebel's families from Friedman's survey (n = 10, 26,
28, 38, 52, 65, 67, 85, 89 at their best-known sides). Sides are swept over (i + j sqrt2)/2, the
values where these constructions go tight; wrapping L layers keeps the excess over floor(s)^2, and
dropping squares covers every smaller n. This beats the grid at 399 of the 1000 puzzles and the
full file scores **23.4405** with the official metric (grid 28.94). The remaining gaps to the
survey for n <= 100 (11, 17, 18, 19, 29, 37, 41, 50, 54, 55, 66, 68-71, 83, 86-88) need other angles.

```bash
export SQUARES_BEST_DIR=/path/to/shared/best   # default: solutions/ in the repo
python -m scripts.build_constructions
python -m scripts.write_submission --out submission.csv
python -m scripts.score_submission submission.csv
```
### Neuroevolution solver and the shared best-per-n store

| Path | What it is |
| --- | --- |
| `wm/best_store.py` | Shared best-per-n store: one JSON per n with the exact decimal strings of the submission. `offer()` keeps a packing only if Kaggle's metric certifies it and it is shorter than the stored one; `submission_rows()` builds the full 1..1000 file |
| `wm/pack_opt.py` | Fast local optimizer: closed-form separating-axis overlap energy and gradient (numba), a penalty method with the box side as a variable (`compact`), and a shrink loop that ends on an exactly feasible packing with a 2e-9 margin (`squeeze`) |
| `wm/neuroevo.py` | A small MLP maps per-square features (grid cell or stored-packing position, rank, Fourier terms, fixed noise, features of n) to an offset and a tilt; CMA-ES evolves its weights. Fitness is Baldwinian: decode, `compact_squeeze`, score the box. The best packing per n is basin-hopped (`refine`) and offered to the store |
| `scripts/evolve_squares.py` | CLI: `--per-n`, one shared net across sizes (default), `--from-store` (deform the stored best instead of the grid), `--baseline` (random restarts with the same budget), `--refine-only` |

```bash
# Point every run, on any machine or thread, at one folder; the store keeps the best certified packing.
export SQUARES_BEST_DIR=/mnt/project-files/squares/best
python -m scripts.evolve_squares --n 17-19 --per-n --gens 30          # ~5 min on 4 cores
python -m scripts.evolve_squares --n 37,50 --per-n --from-store        # refine construction results
```

Parallel runs scale by splitting the n range or the seeds across processes or machines that share
`SQUARES_BEST_DIR`; writes are atomic and the store only ever moves to a shorter certified side.

Results (4 CPU cores, about 30 minutes in total, every packing certified by the official metric):

| n | before (grid or construction) | neuroevolution | best known (Friedman) |
| --- | --- | --- | --- |
| 11 | 3.9142 | 3.87708 | 3.8771 |
| 17 | 4.7071 | 4.67765 | 4.6756 |
| 18 | 4.8284 | 4.82288 | 4.8229 |
| 19 | 4.9142 | 4.88562 | 4.8856 |
| 29 | 6.0000 | 5.99236 | 5.9344 |
| 54 | 7.9142 | 7.85142 | |

No gain at n = 37, 41, 50, 55, 66, 68 within 20-25 generations. Honest control: at n = 5, 10 and
11 random restarts with the same number of `compact_squeeze` calls do about as well as the net (both
reach the optimum for 5 and 10), so for small n the local optimizer does the work. The net earns
its keep from about n = 18 up: at n = 29 random restarts never got below the grid's 6.0, while
the evolved net reached 5.992 twice in separate runs. Deforming a stored construction
(`--from-store`) gave the 17, 18, 19 and 54 results. These packings are in `solutions/`.

### Dimensional relaxation (lifted squares)

`wm/lift.py` gives every square a height so jammed squares can pass over each other, then pushes
the heights back to the plane. Two forms: *slab* (square prisms of thickness 1 in a height box
that shrinks to 0; z is one more separating axis) and *lens* (pair overlap times
1 - |dz|, heights held by a spring that stiffens to infinity). `lift_search` asks for a box a bit
smaller than a stored packing; `lift_compact` is `compact` with lifted squares, for random starts.
`scripts/lift_squares.py` runs each against its flat control on the same seeds:

```bash
python -m scripts.lift_squares random --n 17,29,37 --seeds 16        # lift_compact vs compact
SQUARES_BEST_DIR=/mnt/project-files/squares/best \
  python -m scripts.lift_squares store --n 54 --seeds 8 --tries 60 --shrink 0.0005 --offer
```

Result (2026-10-24, 4 CPUs, about an hour): one store win, n = 54 from 7.85142 to 7.85113
(total score -0.00004). The slab form does nothing: a pair only uses the z axis once it is
almost fully separated in height, so small overlaps never lift. The lens form does lift and
flatten, but on the stored packings at n = 11-150 it nearly always relaxes back into the same jam;
the flat control (jitter + relax) found nothing either. From random starts at n = 10-68 the lens
compaction ties the flat one (better best at 37 and 66, worse at 17 and 26, never below the store).

### Research swarm: composable tools, personas, shared board

`wm/swarm/` turns the solvers above into building blocks that a population of agents with
different theoretical backgrounds combine into new approaches, coordinating through a board.

* **Tools** (`wm/swarm/tools.py`): 20 functions with one signature, `tool(ctx, state, **args) ->
  Packing`, in five kinds: *construct* (grid, store, random, tilted lattice, tilted-block family,
  a larger stored packing with squares dropped, a smaller one wrapped in an L, C4-symmetric seed),
  *perturb* (jolt, relocate into holes, twist field, press, heat), *relax* (FIRE, penalty compact,
  squeeze, 3D lift), *combine* (crossover along a random cut) and *meta* (repeat with Metropolis
  acceptance, best_of). A **recipe** is a JSON list of steps, nested for meta/combine steps:

  ```json
  [{"t": "store"}, {"t": "repeat", "times": 0, "temp": 1e-4,
    "body": [{"t": "relocate", "k": 2, "how": "crowded"}, {"t": "squeeze_box", "rounds": 25}]}]
  ```

  `evaluate(recipe, n, budget)` reruns it with new seeds until the budget is spent and offers the
  best feasible result to the shared store (kept only if Kaggle's metric certifies it and it is
  shorter). `python -m scripts.swarm tools` prints the table.
* **Personas** (`wm/swarm/agents.py`): erdos (combinatorial constructions, monotonicity in n),
  jammer (granular jamming, annealing, lifting), darwin (crossover, recipe mutation), lagrange
  (penalty methods, shaving slack off stored packings), metropolis (basin hopping), escher (C4
  symmetry) and bandit (Thompson sampling over the board's recipes, splicing the best). Each has
  its own recipe grammar and n band; it picks n where the stored packing looks slack next to its
  neighbours' density, avoids sizes others have claimed, and with some probability builds on a
  recipe another agent posted (mutating it or splicing its own tail on, citing the parent).
* **Board** (`wm/swarm/board.py`): one JSON file per post under `$SWARM_BOARD/posts/` (no write
  contention, works on a shared or synced folder). Post kinds: recipe (with parents and the
  reasoning), hypothesis (scored open / supported / refuted from results), claim, result, note.
  `digest.md` summarizes agents, recipe lineage and stats, hypotheses and store wins.

```bash
# The whole roster, one process each, n <= 300, 90 s per recipe run, for two hours
SQUARES_BEST_DIR=/shared/best SWARM_BOARD=/shared/swarm \
  python -m scripts.swarm run --agents erdos,jammer,darwin,lagrange,metropolis:2,homotopy:2,escher,bandit:2 \
  --minutes 120 --max-n 300 --budget 90
python -m scripts.swarm digest                   # board summary
python -m scripts.swarm prompt darwin            # brief for a Claude session acting as darwin
python -m scripts.swarm try --agent me --n 37 --recipe '[{"t":"store"},{"t":"jolt"},{"t":"squeeze_box"}]'
```

A Claude session can join as an agent: `prompt <persona>` gives it the roster entry, the tool
table and the `try` / `note` commands, so it reads the same board and posts recipes the
heuristic agents will pick up and recombine.

### Number theory: which relations between n transfer

`scripts/number_theory_squares.py` asks whether primes, parity, divisibility or factorization
predict how well n packs, once position between consecutive squares is controlled for
(n = k^2 + d, u = d / (2k + 1)). Findings on the shared store (2026-10-25):

* Position is everything. Every n with u < 0.4 beats the grid; above u = 0.5 almost nothing does.
* Prime, even, 4 | n, 3 | n, squarefree, many divisors: residual differences of at most 0.0005
  in score, none significant (prime p = 0.1, the rest p > 0.4). "2 and 4 are similar" holds only
  as s(4m) <= 2 s(m), which never beats the store: small packings waste too much.
* Relations that do transfer are geometric: fixed offset d across k (wrap an L of 2k + 1
  squares, so s((k+1)^2 + d) <= s(k^2 + d) + 1), dropping squares (s monotone), and 2x2
  compositions. `compose_squares --offer` enforces all three; rerun it after any win.
* Tiling j x j copies of an m-packing never wins, and no square packing with m <= 200 is denser
  per unit length than grid columns, so repeating small packings along a strip cannot help.

`wm/strips.py` adds the one family that does win in the grid-only zone: Erdos-Graham style tilted
columns at any angle (not just 45 degrees) in a top band and a right band around an axis-aligned
grid block. Columns of r + 1 squares tilted by theta fill a band r + delta high at density
(r + 1) cos(theta) per unit width, which beats the grid once delta exceeds about sqrt(2 / r), so
it only pays at large k. `python -m scripts.strip_squares --offer` won 133 n in 70-994 (store
23.381 -> 23.304 with the compose cascade), pushing the beats-grid frontier from u ~ 0.45 to ~ 0.5.

## Next steps

1. **v1: RSSM.** Split the latent into deterministic `h` and stochastic `z` with a KL between
   posterior and prior (DreamerV3 style), so the model can represent uncertainty.
2. **Actor-critic in imagination**, then compare its score against the random policy.
3. Harder environments: more balls, occlusion, randomness; then MiniGrid / Crafter / Atari.
