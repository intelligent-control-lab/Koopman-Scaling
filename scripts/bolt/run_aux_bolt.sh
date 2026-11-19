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
# Datasets shipped as portable .npz (name.cache.npz) are converted to the training cache (name.pt) here,
# with the container's numpy, because a .pt pickled by a newer numpy cannot be loaded in this image.
for z in ../data/datasets/*.cache.npz; do
  [ -f "$z" ] || continue; pt="${z%.cache.npz}.pt"; [ -f "$pt" ] && continue
  "$PY" - "$z" "$pt" <<'EOF'
import sys, numpy as np, torch
z = np.load(sys.argv[1])
torch.save({"Ktrain_data": z["Ktrain_data"], "Kval_data": z["Kval_data"], "Ktest_data": z["Ktest_data"],
            "norm_stats": {k[3:]: z[k] for k in z.files if k.startswith("ns_")}}, sys.argv[2])
print("[bolt] built dataset cache", sys.argv[2])
EOF
done
# Limit CPU threads per process so that parallel trainings do not compete for all cores.
NCPU=$(nproc 2>/dev/null || echo 24); P=${PARALLEL:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$(( NCPU / P > 1 ? NCPU / P : 1 ))}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
echo "[bolt] PARALLEL=$P NGPU=${NGPU:-1} OMP_NUM_THREADS=$OMP_NUM_THREADS"
# ---- Preemption tolerance: restore ../log from the Bolt checkpoint, and checkpoint it every CKPT_MIN minutes.
CK="${BOLT_CHECKPOINT_DIR:-}"
if [ -n "$CK" ] && [ -f "$CK/log.tar" ]; then
  echo "[bolt] restoring log/ from checkpoint ($(du -h "$CK/log.tar" | cut -f1))"
  mkdir -p ../log && tar -xf "$CK/log.tar" -C ..
fi
if [ -n "$CK" ]; then
  "$PY" -c "import apple_bolt" 2>/dev/null || "$PY" -m pip install -q apple-bolt --index-url https://pypi.apple.com/simple >/dev/null 2>&1
  ( while sleep $(( ${CKPT_MIN:-15} * 60 + 1 )); do
      tar -cf "$CK/log.tar.tmp" -C .. log 2>/dev/null && mv "$CK/log.tar.tmp" "$CK/log.tar" \
        && "$PY" -c "import apple_bolt; apple_bolt.checkpoint()" >/dev/null 2>&1 \
        && echo "[bolt] checkpoint saved $(date +%H:%M)"
    done ) > /dev/null 2>&1 &
  CKPT_PID=$!
fi
sync_out() { mkdir -p "$ART"; cp -r ../log/aux_tuning_* "$ART"/ 2>/dev/null; for d in ../log/aux_*_"$ENV"_*; do [ -d "$d" ] && mkdir -p "$ART/runs" && cp -r "$d" "$ART/runs/"; done; }
trap '[ -n "${CKPT_PID:-}" ] && kill $CKPT_PID 2>/dev/null; sync_out' EXIT
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
