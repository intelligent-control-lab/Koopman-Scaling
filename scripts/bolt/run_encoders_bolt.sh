#!/bin/bash
# Bolt entry point: train the damped-pendulum encoders for the fixed-feature study (Fig. 5),
# for latent multipliers MULTS (default "2 8 32") and the five seeds, 4000 windows each, as split_study.py.
set -uo pipefail
cd "$(dirname "$0")/.."   # -> scripts/
ART="${ARTIFACT_DIR:-/mnt/task_wrapper/user_output/artifacts}"
PY=""
for P in /workspace/isaaclab/_isaac_sim/python.sh python3 python; do
  command -v "$P" >/dev/null 2>&1 || [ -x "$P" ] || continue
  "$P" -c "import torch, pandas, yaml, tqdm, scipy" >/dev/null 2>&1 && { PY="$P"; break; }
done
[ -n "$PY" ] || { echo "[bolt] no python"; exit 2; }
echo "[bolt] python: $PY"
for z in ../data/datasets/*.cache.npz; do
  pt="${z%.cache.npz}.pt"; [ -f "$pt" ] && continue
  "$PY" - "$z" "$pt" <<'PYEOF'
import sys, numpy as np, torch
z = np.load(sys.argv[1]); ns = {k[3:]: z[k] for k in z.files if k.startswith("ns_")}
torch.save({"Ktrain_data": z["Ktrain_data"], "Kval_data": z["Kval_data"], "Ktest_data": z["Ktest_data"],
            "norm_stats": ns or None}, sys.argv[2]); print("[bolt] built", sys.argv[2])
PYEOF
done
MULTS=${MULTS:-"2 8 32"}; NGPU=${NGPU:-2}; PARALLEL=${PARALLEL:-8}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
trap 'mkdir -p "$ART/runs"; for d in ../log/page_grid_m4000_k* ../log/split_s*_m4000_k*; do [ -d "$d" ] && cp -r "$d" "$ART/runs/"; done' EXIT
k=0
for mult in $MULTS; do for seed in 17382 76849 20965 84902 51194; do
  CUDA_VISIBLE_DEVICES=$((k % NGPU)) "$PY" train_one_encoder.py $seed $mult > ../log/enc_${seed}_${mult}.log 2>&1 &
  k=$((k + 1)); while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do sleep 5; done
done; done
wait
grep -h "\[encoder\] done" ../log/enc_*.log | sort; ls -d ../log/*_m4000_k* | wc -l
