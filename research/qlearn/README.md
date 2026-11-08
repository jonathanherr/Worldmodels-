# Q-learning over macro moves (2026-11-08)

`qpack.py`: fixed box at L* (10% of the way from our side to the Kaggle leader's), record scaled in and
relaxed (positive overlap energy). Linear Q over (state, move) features picks macro moves on the 12 most
overlapping squares (kick, turn, relocate to hole, 7-square cluster turn, swap with other tilt), each
followed by a FIRE relax. Reward = drop in log energy, +10 for reaching zero (a feasible, better packing).
`run1.sh` compares q / random / greedy (best of 6 tries per step) at equal env steps, 12 min each.

Result (n 147, 172, 204; 14k-28k steps each): 0 feasible. Best energies 0.22-0.86 of the FIRE start,
which is what a deeper L-BFGS relax of the start already gives: no policy found a lower basin.
Q did learn (episode end energy ~1e-4.6 vs random ~1e-2.3; return -3.7 vs -9.2) but what it learned is
"make gentle moves"; returns stayed negative, i.e. no multi-step move sequence descends below the record basin.
Raw: /mnt/project-files/squares/qlearn/r1_*.jsonl
