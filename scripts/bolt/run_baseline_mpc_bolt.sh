#!/bin/bash
# Bolt entry point: closed-loop G1/Go2 tracking of the EDMD and NNDM baselines (Table 5 of the paper).
#   STAGE=smoke : one EDMD and one NNDM model on 20 test trajectories (checks image, paths, and run time)
#   STAGE=full  : EDMD selects the MPC input weight r in {0, 1e-3} on validation trajectories 1000-1499
#                 (the protocol of the tuned Koopman models), then all models track test trajectories 0-999.
#                 NNDM uses random shooting (1000 samples) and has no r. EDMD runs on GPU 0, NNDM on GPU 1.
# Inputs (in the tar): ../baseline_models/<ENV>/{edmd,mlp}_*.{npz,pt}, ../data/norm/<ENV>_norm.npz, reference npz.
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
echo "[bolt] python: $PY ENV=$ENV STAGE=$STAGE"
ROOT=$(realpath ..); OUT=$ROOT/log/baseline_mpc_$ENV; mkdir -p "$OUT"
NORM=$ROOT/data/norm/${ENV}_norm.npz
trap 'mkdir -p "$ART"; cp -r "$OUT" "$ART"/' EXIT
rows() { [ -f "$1" ] && $PY -c "import pandas as pd,sys; print(len(pd.read_csv(sys.argv[1])))" "$1" || echo 0; }
models_csv() {  # method ext out.csv [limit]
  ls "$ROOT"/baseline_models/$ENV/$1_${ENV}_*.${2} | head -n "${4:-100}" \
    | awk -v e="$ENV" 'BEGIN{print "env_name,model_path"} {print e","$0}' > "$3"
}
track() {  # controller models_csv out_csv traj_start num_trajs r gpu timeout: one model per Isaac process
  local total before fails=0; total=$(rows "$2")
  while :; do
    before=$(rows "$3"); [ "$before" -ge "$total" ] && break
    (cd ../control && CUDA_VISIBLE_DEVICES=$7 timeout "$8" $PY mpc_tracking.py --headless --controller "$1" \
       --dataset_path "$NORM" --csv_log_path "$2" --save_path "$3" --traj_start "$4" --num_trajs "$5" \
       --mpc_r "$6" --max_models 1) || true
    if [ "$(rows "$3")" -le "$before" ]; then
      fails=$((fails + 1)); echo "[track] no progress ($1, attempt $fails)"
      [ $fails -ge 2 ] && { echo "[track] giving up on $1 after $before/$total models"; break; }
    else fails=0; fi
  done
}
if [ "$STAGE" = smoke ]; then
  models_csv edmd npz "$OUT/smoke_edmd_models.csv" 1; models_csv mlp pt "$OUT/smoke_nndm_models.csv" 1
  track edmd "$OUT/smoke_edmd_models.csv" "$OUT/smoke_edmd.csv" 0 20 0.001 0 1200 &
  track nndm "$OUT/smoke_nndm_models.csv" "$OUT/smoke_nndm.csv" 0 20 0 1 1800 &
  wait; cat "$OUT"/smoke_*.csv; exit 0
fi
models_csv edmd npz "$OUT/edmd_models.csv" 1; models_csv mlp pt "$OUT/nndm_models.csv"  # EDMD is deterministic: one model
(
  for r in 0 0.001; do track edmd "$OUT/edmd_models.csv" "$OUT/edmd_val_r$r.csv" 1000 500 $r 0 1800; done
  R=$($PY - "$OUT" <<'EOF'
import sys, pandas as pd
o = sys.argv[1]; s = {r: pd.read_csv(f"{o}/edmd_val_r{r}.csv").mean_survival_steps.mean() for r in ("0", "0.001")}
print(max(s, key=s.get)); print(s, file=sys.stderr)
EOF
)
  echo "[edmd] selected r=$R on validation"; echo "$R" > "$OUT/edmd_selected_r.txt"
  track edmd "$OUT/edmd_models.csv" "$OUT/edmd_test.csv" 0 1000 "$R" 0 1800
) &
track nndm "$OUT/nndm_models.csv" "$OUT/nndm_test.csv" 0 1000 0 1 7200 &
wait
echo "[done]"; for f in "$OUT"/*_test.csv; do echo "$f"; cat "$f"; done
