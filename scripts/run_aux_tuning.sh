#!/bin/bash
# Validation-based tuning of the auxiliary-loss weights for the G1/Go2 closed-loop study.
#
# Protocol (both arms get the same tuning budget; nothing is selected on the test trajectories):
#   Stage 1 (select): m = 140k windows, n_mult in {1,4,16}, 2 seeds.
#     baseline : train_steps in {20k, 60k}
#     +Aux    : train_steps in {20k, 60k} x ctrl_loss_weight in {0.01, 0.1, 1} x cov_loss_weight in {0.1, 1}
#     every model is evaluated with MPC input weight r in {0, 1e-3} on VALIDATION reference
#     trajectories 1000-1499, which the paper never uses.
#     select_aux_config.py picks, per arm, the setting with the highest mean validation survival.
#   Stage 2 (final): the selected settings for Baseline, +Cov, +Ctrl, +Aux (+Cov/+Ctrl use the selected
#     weight of their own loss), full grid m x n_mult x 5 seeds, evaluated on the paper's TEST
#     trajectories 0-999. Report every run.
#
# Usage (Linux + NVIDIA GPU + Isaac Lab; G1/Go2 data under data/):
#   ENV=G1 bash run_aux_tuning.sh select     # stage 1: train + validation MPC
#   python select_aux_config.py --env G1     # merges the shard CSVs, writes ../log/aux_tuning_<ENV>/selected.json
# Optional: PARALLEL=k concurrent trainings; SHARD=i NSHARDS=k to split a stage over k tasks.
#   ENV=G1 bash run_aux_tuning.sh final      # stage 2: train + test MPC
# Set PY (training) and ISAAC_PY (MPC) to the right interpreters if needed (e.g. ISAAC_PY="/path/isaaclab.sh -p").
set -euo pipefail
cd "$(dirname "$0")"
ENV=${ENV:-G1}
STAGE=${1:?usage: run_aux_tuning.sh select|final}
PY=${PY:-python}
ISAAC_PY=${ISAAC_PY:-$PY}
ROOT=../log/aux_tuning_${ENV}
mkdir -p "$ROOT"

train() {  # project m n_mult seed steps use_cov use_ctrl w_cov w_ctrl [gpu]
  local flags="" gpu=${10:-0}
  [ "$6" = 1 ] && flags="$flags --use_covariance_loss"
  [ "$7" = 1 ] && flags="$flags --use_control_loss"
  CUDA_VISIBLE_DEVICES=$gpu $PY train_model.py --project_name "$1" --env_name "$ENV" --sample_size "$2" --encode_dim "$3" \
    --layer_depth 3 --hidden_dim 256 --seed "$4" --train_steps "$5" \
    --cov_loss_weight "$8" --ctrl_loss_weight "$9" --use_residual --multiply_encode_by_input_dim $flags \
    > "../log/$1.train.txt" 2>&1 || echo "[train] FAILED: $1 (see log/$1.train.txt)"
}

# Run up to PARALLEL trainings at once; each run has its own project directory (no shared log file).
# NGPU > 1 spreads trainings and MPC chunks over the GPUs of one node.
PARALLEL=${PARALLEL:-1}; NGPU=${NGPU:-1}
# Training-set sizes: M_SELECT for stage 1, M_GRID for stage 2 (override when a dataset has fewer windows).
M_SELECT=${M_SELECT:-140000}; M_GRID=${M_GRID:-"1000 4000 16000 64000 140000"}
# SHARD/NSHARDS split the runs of a stage across Bolt tasks (run index modulo NSHARDS).
SHARD=${SHARD:-0}; NSHARDS=${NSHARDS:-1}; RUN_IDX=0; MY_IDX=0
MY_PROJECTS=()
queue() {  # project_prefix m n_mult seed steps use_cov use_ctrl w_cov w_ctrl
  local p="$1__m$2_nm$3_s$4"
  RUN_IDX=$((RUN_IDX + 1))
  [ $(((RUN_IDX - 1) % NSHARDS)) -eq "$SHARD" ] || return 0
  MY_PROJECTS+=("$p")
  MY_IDX=$((MY_IDX + 1))
  already_trained "$p" "$2" "$3" "$4" && return 0
  while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do sleep 5; done
  echo "[train] $p (gpu $(( MY_IDX % NGPU )))"
  train "$p" "${@:2}" $(( MY_IDX % NGPU )) &
}

already_trained() {  # project m n_mult seed
  local log=../log/$1/koopman_results_log.csv
  [ -f "$log" ] && $PY - "$log" "$2" "$3" "$4" <<'EOF'
import sys, pandas as pd
d = pd.read_csv(sys.argv[1])
sys.exit(0 if ((d.train_samples == int(sys.argv[2])) & (d.encode_dim_param == int(sys.argv[3]))
               & (d.seed == int(sys.argv[4]))).any() else 1)
EOF
}

rows() { [ -f "$1" ] && $PY -c "import pandas as pd,sys; print(len(pd.read_csv(sys.argv[1])))" "$1" 2>/dev/null || echo 0; }
mpc_chunk() {  # models_csv out_csv traj_start num_trajs r gpu: one model per Isaac process, each with a time limit
  local total done before fails=0
  total=$(rows "$1")
  while :; do
    before=$(rows "$2")
    [ "$before" -ge "$total" ] && break
    (cd ../control && CUDA_VISIBLE_DEVICES=$6 timeout ${MPC_TIMEOUT:-1200} $ISAAC_PY mpc_tracking.py --headless \
       --csv_log_path "$1" --save_path "$2" --traj_start "$3" --num_trajs "$4" --mpc_r "$5" --max_models 1) || true
    done=$(rows "$2")
    if [ "$done" -le "$before" ]; then
      fails=$((fails + 1)); echo "[mpc] no progress on attempt $fails"
      if [ $fails -ge 2 ]; then   # record the next model as failed and move on
        $PY - "$1" "$2" "$3" "$4" "$5" <<'EOF'
import sys, os, pandas as pd
m = pd.read_csv(sys.argv[1]); out = sys.argv[2]
done = set(pd.read_csv(out).model_path) if os.path.exists(out) else set()
todo = [p for p in m.model_path if p not in done]
if todo:
    row = pd.DataFrame([dict(model_path=todo[0], robot=m.env_name.iloc[0], traj_start=int(sys.argv[3]),
                             num_trajs=int(sys.argv[4]), mpc_r=float(sys.argv[5]), failed=1)])
    (pd.concat([pd.read_csv(out), row], ignore_index=True) if os.path.exists(out) else row).to_csv(out, index=False)
    print("[mpc] recorded as failed:", todo[0])
EOF
        fails=0
      fi
    else fails=0; fi
  done
}
run_mpc() {  # models_csv out_csv traj_start r  (with NGPU > 1: one chunk of models per GPU, in parallel)
  local nt; nt=$([ "$3" = 0 ] && echo 1000 || echo 500)
  if [ "$NGPU" -le 1 ]; then
    mpc_chunk "$1" "$2" "$3" "$nt" "$4" 0
    return
  fi
  $PY - "$1" "$NGPU" <<'EOF'
import sys, pandas as pd
d = pd.read_csv(sys.argv[1]); k = int(sys.argv[2])
for g in range(k):
    d.iloc[g::k].to_csv(f"{sys.argv[1]}.chunk{g}", index=False)
EOF
  for g in $(seq 0 $((NGPU - 1))); do
    mpc_chunk "$1.chunk$g" "$2.chunk$g" "$3" "$nt" "$4" "$g" > "$2.chunk$g.log" 2>&1 &
  done
  wait
  $PY - "$2" "$NGPU" <<'EOF'
import sys, os, pandas as pd
parts = [f"{sys.argv[1]}.chunk{g}" for g in range(int(sys.argv[2]))]
pd.concat([pd.read_csv(p) for p in parts if os.path.exists(p)], ignore_index=True).to_csv(sys.argv[1], index=False)
EOF
}

collect() {  # out_csv project...
  local out=$1; shift
  $PY - "$out" "$@" <<'EOF'
import sys, os, pandas as pd
frames = [pd.read_csv(f"../log/{p}/koopman_results_log.csv").assign(project=p.split("__")[0])
          for p in sys.argv[2:] if os.path.exists(f"../log/{p}/koopman_results_log.csv")]
pd.concat(frames, ignore_index=True).to_csv(sys.argv[1], index=False)
EOF
}

if [ "$STAGE" = select ]; then
  projects=()
  for steps in 20000 60000; do
    p=aux_sel_${ENV}_base_s${steps}; projects+=("$p")
    for wctrl in 0.01 0.1 1; do for wcov in 0.1 1; do
      projects+=("aux_sel_${ENV}_both_s${steps}_c${wcov}_u${wctrl}")
    done; done
  done
  for p in "${projects[@]}"; do
    steps=$(echo "$p" | sed -E 's/.*_s([0-9]+).*/\1/')
    for nm in 1 4 16; do for seed in 17382 76849; do
      if [[ $p == *_base_* ]]; then
        queue "$p" "$M_SELECT" "$nm" "$seed" "$steps" 0 0 1 1
      else
        wcov=$(echo "$p" | sed -E 's/.*_c([0-9.]+)_u.*/\1/'); wctrl=$(echo "$p" | sed -E 's/.*_u([0-9.]+)$/\1/')
        queue "$p" "$M_SELECT" "$nm" "$seed" "$steps" 1 1 "$wcov" "$wctrl"
      fi
    done; done
  done
  wait
  collect "$ROOT/select_models_${SHARD}.csv" "${MY_PROJECTS[@]}"
  for r in 0 0.001; do run_mpc "$(realpath "$ROOT/select_models_${SHARD}.csv")" "$(realpath "$ROOT")/select_val_r${r}_${SHARD}.csv" 1000 "$r"; done
  echo "Now run: python select_aux_config.py --env $ENV"
elif [ "$STAGE" = final ]; then
  SEL=$ROOT/selected.json
  [ -f "$SEL" ] || { echo "missing $SEL (run select_aux_config.py)"; exit 1; }
  read -r BSTEPS ASTEPS WCOV WCTRL R < <($PY -c "
import json; s = json.load(open('$SEL'))
print(s['baseline']['train_steps'], s['both']['train_steps'], s['both']['cov_loss_weight'],
      s['both']['ctrl_loss_weight'], s['mpc_r'])")
  projects=()
  for arm in base cov ctrl both; do
    p=aux_final_${ENV}_${arm}; projects+=("$p")
    for m in $M_GRID; do for nm in 1 2 4 8 16; do
      for seed in 17382 76849 20965 84902 51194; do
        case $arm in
          base) queue "$p" "$m" "$nm" "$seed" "$BSTEPS" 0 0 1 1 ;;
          cov)  queue "$p" "$m" "$nm" "$seed" "$ASTEPS" 1 0 "$WCOV" 1 ;;
          ctrl) queue "$p" "$m" "$nm" "$seed" "$ASTEPS" 0 1 1 "$WCTRL" ;;
          both) queue "$p" "$m" "$nm" "$seed" "$ASTEPS" 1 1 "$WCOV" "$WCTRL" ;;
        esac
      done
    done; done
  done
  wait
  collect "$ROOT/final_models_${SHARD}.csv" "${MY_PROJECTS[@]}"
  run_mpc "$(realpath "$ROOT/final_models_${SHARD}.csv")" "$(realpath "$ROOT")/final_test_${SHARD}.csv" 0 "$R"
  echo "Test results: $ROOT/final_test_${SHARD}.csv"
else
  echo "unknown stage $STAGE"; exit 1
fi
