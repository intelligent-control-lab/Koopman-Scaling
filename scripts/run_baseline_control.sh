#!/bin/bash
# Closed-loop G1/Go2 tracking of the EDMD and NNDM baselines (Table 5 of the paper), in Isaac Lab.
#
#   EDMD uses the same linear MPC as the Koopman models and selects the MPC input weight r in {0, 1e-3} on the
#   validation reference trajectories 1000-1499; NNDM uses random-shooting MPC (1,000 samples). Both then track
#   the test reference trajectories 0-999. Each model runs in its own Isaac process.
#
# Inputs: ../baseline_models/<ENV>/{edmd,mlp}_<ENV>_*.{npz,pt} (from baseline_comparison.py) and
#         ../data/norm/<ENV>_norm.npz (normalization statistics of the training data).
# Usage (from scripts/):  ENV=G1 ISAAC_PY="/path/to/isaaclab.sh -p" bash run_baseline_control.sh
set -uo pipefail
cd "$(dirname "$0")"
ENV=${ENV:-G1}; PY=${PY:-python}; ISAAC_PY=${ISAAC_PY:-$PY}
ROOT=$(cd .. && pwd); OUT=$ROOT/log/baseline_mpc_$ENV; mkdir -p "$OUT"
NORM=$ROOT/data/norm/${ENV}_norm.npz
rows() { [ -f "$1" ] && $PY -c "import pandas as pd, sys; print(len(pd.read_csv(sys.argv[1])))" "$1" || echo 0; }
models_csv() {  # method extension out.csv [limit]
  ls "$ROOT"/baseline_models/$ENV/$1_${ENV}_*.$2 | head -n "${4:-100}" \
    | awk -v e="$ENV" 'BEGIN{print "env_name,model_path"} {print e","$0}' > "$3"
}
track() {  # controller models.csv out.csv traj_start num_trajs r
  local total before fails=0; total=$(rows "$2")
  while :; do
    before=$(rows "$3"); [ "$before" -ge "$total" ] && break
    (cd ../control && $ISAAC_PY mpc_tracking.py --headless --controller "$1" --dataset_path "$NORM" \
       --csv_log_path "$2" --save_path "$3" --traj_start "$4" --num_trajs "$5" --mpc_r "$6" --max_models 1) || true
    if [ "$(rows "$3")" -le "$before" ]; then
      fails=$((fails + 1)); [ $fails -ge 2 ] && { echo "[track] stopping $1 after $before/$total models"; break; }
    else fails=0; fi
  done
}
models_csv edmd npz "$OUT/edmd_models.csv" 1     # EDMD is deterministic: one model
models_csv mlp pt "$OUT/nndm_models.csv"
for r in 0 0.001; do track edmd "$OUT/edmd_models.csv" "$OUT/edmd_val_r$r.csv" 1000 500 $r; done
R=$($PY - "$OUT" <<'PYEOF'
import sys, pandas as pd
o = sys.argv[1]
s = {r: pd.read_csv(f"{o}/edmd_val_r{r}.csv").mean_survival_steps.mean() for r in ("0", "0.001")}
print(max(s, key=s.get))
PYEOF
)
echo "[edmd] r selected on validation: $R"
track edmd "$OUT/edmd_models.csv" "$OUT/edmd_test.csv" 0 1000 "$R"
track nndm "$OUT/nndm_models.csv" "$OUT/nndm_test.csv" 0 1000 0
