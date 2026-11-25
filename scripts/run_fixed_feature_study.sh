#!/bin/bash
# Fixed-feature test of the sampling theory on the damped pendulum (Section 6.1.4, Fig. 5 and Table 3 of the paper).
#
#   1. train the frozen encoders: five seeds x latent multipliers MULTS, 4,000 trajectory segments each;
#   2. simulate an independent reference pool (16,000 trajectories) for the population quantities;
#   3. fit (A, B) by least squares on m independent transitions and record the sampling / projection errors
#      and the event ||G_tilde - I|| <= 1/2 used in the proof of Proposition 1 (log/fixed_feature_study.csv);
#   4. fit the empirical sample-threshold laws (censored likelihood, leave-one-seed-out prediction);
#   5. plot the three panels of Fig. 5.
#
# Usage (from scripts/):  bash run_fixed_feature_study.sh            Options: PY=python PARALLEL=4 NGPU=1
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}; PARALLEL=${PARALLEL:-4}; NGPU=${NGPU:-1}
MULTS=${MULTS:-"1 2 4 8 16 32"}
SEEDS="17382 76849 20965 84902 51194"
mkdir -p ../log
k=0
for mult in $MULTS; do
  for seed in $SEEDS; do
    CUDA_VISIBLE_DEVICES=$((k % NGPU)) $PY train_one_encoder.py "$seed" "$mult" > "../log/encoder_${seed}_${mult}.txt" 2>&1 &
    k=$((k + 1))
    while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do sleep 5; done
  done
done
wait
[ -f ../log/ffs_reference_pool.npz ] || $PY make_reference_pool.py ../log/ffs_reference_pool.npz
MULTS="$MULTS" FFS_OUT=fixed_feature_study.csv $PY fixed_feature_study.py
FFS_CSV=fixed_feature_study.csv $PY threshold_law_fits.py
FFS_CSV=fixed_feature_study.csv $PY plot_fixed_feature_study.py ../log/fixed_feature_study.pdf
