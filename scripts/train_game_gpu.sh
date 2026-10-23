#!/usr/bin/env bash
# Bigger game-world run for a single CUDA GPU (12+ GB). About 1-2 hours on an RTX 3090/4090.
# Writes runs/game_gpu/, then evaluates it and exports the browser demo to web_gpu/.
set -euo pipefail
cd "$(dirname "$0")/.."

[ -f data/game_large/train.npz ] || python -m scripts.collect_data --env game --train-steps 300000 --test-steps 15000 --out data/game_large
python -m scripts.train_v0 --data data/game_large/train.npz --out runs/game_gpu \
  --steps 30000 --ch 32 --batch 32 --latent 64 --hidden 512 --lr 3e-4
python -m scripts.eval_game --run runs/game_gpu --data data/game_large/test.npz
python -m scripts.export_web --run runs/game_gpu --data data/game_large/test.npz --out web_gpu
cp web/index.html web/dream.js web_gpu/
cp runs/game_gpu/long_dream.gif runs/game_gpu/rollout_error.png web_gpu/
echo "Open the demo with: python -m http.server -d web_gpu 8000  ->  http://localhost:8000"
