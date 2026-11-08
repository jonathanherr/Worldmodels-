#!/bin/bash
# round 1: frac 0.1 of gap to leader, 3 policies x n
C=/home/user/qwork/submission_v10_20261105.csv
O=/mnt/project-files/squares/qlearn
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1
run(){ python3 qpack.py $C $1 --target $2 --policy $3 --minutes $4 --restart-best --seed 0 --out $O/r1_$1_$3.jsonl > $O/r1_$1_$3.log 2>&1; }
M=${M:-12}
run 147 12.656273 q $M & run 147 12.656273 random $M & run 147 12.656273 greedy $M & run 172 13.617611 q $M & wait
run 172 13.617611 random $M & run 172 13.617611 greedy $M & run 204 14.807214 q $M & run 204 14.807214 random $M & wait
run 204 14.807214 greedy $M & wait
