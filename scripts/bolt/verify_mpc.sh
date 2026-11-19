#!/bin/bash
# Train one small G1 model, then track the same 20 reference trajectories with the legacy and the vectorized
# mpc_tracking.py, and print both results (they should agree up to simulator nondeterminism).
set -uo pipefail
cd "$(dirname "$0")/.."
ART="${ARTIFACT_DIR:-/mnt/task_wrapper/user_output/artifacts}"; mkdir -p "$ART" ../log
PY=/workspace/isaaclab/_isaac_sim/python.sh
# build the dataset cache with the container's numpy (the slim data ships as a portable .npz)
$PY - <<'EOF'
import numpy as np, torch
z = np.load("../data/datasets/g1_slim.npz")
torch.save({"Ktrain_data": z["Ktrain_data"], "Kval_data": z["Kval_data"], "Ktest_data": z["Ktest_data"],
            "norm_stats": {k[3:]: z[k] for k in z.files if k.startswith("ns_")}},
           "../data/datasets/dataset_G1_norm_Ktrain_140000_Kval_20000_Ktest_20000_Ksteps_15.pt")
EOF
$PY train_model.py --project_name verify_G1 --env_name G1 --sample_size 1000 --encode_dim 4 --layer_depth 3 \
  --hidden_dim 256 --seed 17382 --train_steps 3000 --use_residual --multiply_encode_by_input_dim --use_control_loss > ../log/verify_train.txt 2>&1 || { tail -30 ../log/verify_train.txt; exit 3; }
M=$(realpath ../log/verify_G1/koopman_results_log.csv)
for v in legacy new; do
  f=mpc_tracking.py; [ $v = legacy ] && f=mpc_tracking_legacy.py
  for r in 0 0.001; do
    t0=$(date +%s)
    (cd ../control && $PY $f --headless --csv_log_path "$M" --save_path "$ART/verify_${v}_r$r.csv" \
       --traj_start 0 --num_trajs 20 --mpc_r $r > "$ART/verify_${v}_r$r.log" 2>&1)
    echo "$v r=$r wall $(( $(date +%s) - t0 )) s"
  done
done
for f in "$ART"/verify_*.csv; do echo "== $(basename $f)"; cut -d, -f6-14 "$f"; done
