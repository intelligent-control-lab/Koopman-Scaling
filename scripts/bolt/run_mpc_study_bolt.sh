#!/bin/bash
# Bolt entry point for two closed-loop studies on G1/Go2 (Isaac Lab), results in artifacts/mpc_study_<ENV>/.
#   STAGE=smoke   : 20 test trajectories, one Koopman (n_mult=16) and one NNDM model, planners cem and grad
#   STAGE=clip    : closed-form linear MPC (r=1e-3) for the tuned Koopman models at m=140k (base and +Both, all
#                   latent sizes, five seeds) and for EDMD (validation-selected r), with input-clipping statistics
#   STAGE=planner : planner-matched comparison at n_mult=16: Koopman base, Koopman +Both, and NNDM (three seeds each),
#                   each with random shooting (rs), CEM, and gradient MPC (grad), all minimizing the same objective
set -uo pipefail
cd "$(dirname "$0")/.."   # -> scripts/
ENV=${ENV:-G1}; STAGE=${STAGE:-smoke}; R=${MPC_R:-0.001}; EDMD_R=${EDMD_R:-0.001}
ART="${ARTIFACT_DIR:-/mnt/task_wrapper/user_output/artifacts}"
PY=""
for P in /workspace/isaaclab/_isaac_sim/python.sh /workspace/isaaclab/_isaac_sim/kit/python/bin/python3 python3 python; do
  command -v "$P" >/dev/null 2>&1 || [ -x "$P" ] || continue
  "$P" -c "import torch, pandas, yaml, tqdm" >/dev/null 2>&1 && { PY="$P"; break; }
done
[ -n "$PY" ] || { echo "[bolt] no python"; exit 2; }
echo "[bolt] python: $PY ENV=$ENV STAGE=$STAGE"
ROOT=$(realpath ..); OUT=$ROOT/log/mpc_study_$ENV; mkdir -p "$OUT"; NORM=$ROOT/data/norm/${ENV}_norm.npz
trap 'mkdir -p "$ART"; cp -r "$OUT" "$ART"/' EXIT
rows() { [ -f "$1" ] && $PY -c "import pandas as pd,sys; print(len(pd.read_csv(sys.argv[1])))" "$1" || echo 0; }
mk() { echo "env_name,model_path" > "$1"; shift; for f in "$@"; do echo "$ENV,$f" >> "$1.tmp"; done; cat "$1.tmp" >> "$1"; rm -f "$1.tmp"; }
track() {  # models_csv out_csv traj_start num_trajs gpu timeout args...: one model per Isaac process
  local m=$1 o=$2 t0=$3 nt=$4 g=$5 to=$6; shift 6; local total before fails=0; total=$(rows "$m")
  while :; do
    before=$(rows "$o"); [ "$before" -ge "$total" ] && break
    (cd ../control && CUDA_VISIBLE_DEVICES=$g timeout "$to" $PY mpc_tracking.py --headless --dataset_path "$NORM" \
       --csv_log_path "$m" --save_path "$o" --traj_start "$t0" --num_trajs "$nt" --max_models 1 "$@") || true
    if [ "$(rows "$o")" -le "$before" ]; then
      fails=$((fails + 1)); echo "[track] no progress on $o (attempt $fails)"
      [ $fails -ge 2 ] && { echo "[track] giving up on $o at $before/$total"; break; }
    else fails=0; fi
  done
}
KM=$ROOT/koopman_models/$ENV; BM=$ROOT/baseline_models/$ENV
K16=$(ls $KM/*__m140000_nm16_s{17382,76849,20965}.pth); N3=$(ls $BM/mlp_${ENV}_*.pt)
if [ "$STAGE" = smoke ]; then
  mk "$OUT/sm_k.csv" $(ls $KM/*_base__m140000_nm16_s17382.pth); mk "$OUT/sm_n.csv" $(ls $BM/mlp_${ENV}_17382_*.pt)
  for p in cem grad; do
    track "$OUT/sm_k.csv" "$OUT/smoke_koopman_$p.csv" 0 20 0 1800 --controller koopman --planner $p --mpc_r $R &
    track "$OUT/sm_n.csv" "$OUT/smoke_nndm_$p.csv" 0 20 1 1800 --controller nndm --planner $p --mpc_r $R &
    wait
  done
  tail -n +1 "$OUT"/smoke_*.csv; exit 0
fi
if [ "$STAGE" = clip ]; then
  ALL=( $(ls $KM/*__m140000_nm*_s*.pth) ); H=$(( ${#ALL[@]} / 2 ))
  mk "$OUT/clip_a.csv" "${ALL[@]:0:$H}"; mk "$OUT/clip_b.csv" "${ALL[@]:$H}"
  mk "$OUT/clip_e.csv" $(ls $BM/edmd_${ENV}_17382_*.npz)
  ( track "$OUT/clip_a.csv" "$OUT/clip_koopman_a.csv" 0 1000 0 1800 --controller koopman --mpc_r $R
    track "$OUT/clip_e.csv" "$OUT/clip_edmd.csv" 0 1000 0 1800 --controller edmd --mpc_r $EDMD_R ) &
  track "$OUT/clip_b.csv" "$OUT/clip_koopman_b.csv" 0 1000 1 1800 --controller koopman --mpc_r $R &
  wait; echo "[done]"; exit 0
fi
mk "$OUT/pl_k.csv" $K16; mk "$OUT/pl_n.csv" $N3
for p in rs cem grad; do
  track "$OUT/pl_k.csv" "$OUT/planner_koopman_$p.csv" 0 1000 0 7200 --controller koopman --planner $p --mpc_r $R &
  track "$OUT/pl_n.csv" "$OUT/planner_nndm_$p.csv" 0 1000 1 7200 --controller nndm --planner $p --mpc_r $R &
  wait
done
echo "[done]"
