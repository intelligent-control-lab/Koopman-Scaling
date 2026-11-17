#!/bin/bash
# Bolt entry point for the G1/Go2 auxiliary-loss tuning (see scripts/run_aux_tuning.sh).
#   STAGE=smoke  : train one small model and track 20 validation trajectories (checks image + paths)
#   STAGE=select : stage 1 of run_aux_tuning.sh;  STAGE=final : stage 2 (needs log/aux_tuning_<ENV>/selected.json)
# Results (CSVs, training logs, checkpoints) are copied to the Bolt artifact directory.
set -uo pipefail
cd "$(dirname "$0")/.."   # -> scripts/
ENV=${ENV:-G1}; STAGE=${STAGE:-smoke}
ART="${ARTIFACT_DIR:-/mnt/task_wrapper/user_output/artifacts}"
PY=""
for P in /workspace/isaaclab/_isaac_sim/python.sh /workspace/isaaclab/_isaac_sim/kit/python/bin/python3 python3 python; do
  command -v "$P" >/dev/null 2>&1 || [ -x "$P" ] || continue
  "$P" -c "import torch, pandas, yaml, tqdm" >/dev/null 2>&1 && { PY="$P"; break; }
done
[ -n "$PY" ] || { echo "[bolt] no python with torch/pandas/yaml/tqdm"; exit 2; }
echo "[bolt] python: $PY"; "$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
"$PY" -c "import isaaclab, isaaclab_tasks; print('isaaclab ok')" || echo "[bolt] WARNING: isaaclab import failed"
export PY ISAAC_PY="$PY" ENV
sync_out() { mkdir -p "$ART"; cp -r ../log/aux_tuning_* "$ART"/ 2>/dev/null; for d in ../log/aux_*_"$ENV"_*; do [ -d "$d" ] && mkdir -p "$ART/runs" && cp -r "$d" "$ART/runs/"; done; }
trap sync_out EXIT
if [ "$STAGE" = smoke ]; then
  P=aux_smoke_${ENV}
  "$PY" train_model.py --project_name $P --env_name "$ENV" --sample_size 1000 --encode_dim 1 --layer_depth 3 \
    --hidden_dim 256 --seed 17382 --train_steps 2000 --use_residual --multiply_encode_by_input_dim \
    --use_control_loss --use_covariance_loss || exit 3
  mkdir -p ../log/aux_tuning_$ENV
  cp ../log/$P/koopman_results_log.csv ../log/aux_tuning_$ENV/smoke_models.csv
  M=$(realpath ../log/aux_tuning_$ENV/smoke_models.csv); O=$(realpath ../log/aux_tuning_$ENV)/smoke_val.csv
  (cd ../control && time "$ISAAC_PY" mpc_tracking.py --headless --csv_log_path "$M" --save_path "$O" \
     --traj_start 1000 --num_trajs 20 --mpc_r 0) || exit 4
  cat "$O"; cp -r ../log/$P "$ART"/ 2>/dev/null
else
  bash run_aux_tuning.sh "$STAGE"
fi
