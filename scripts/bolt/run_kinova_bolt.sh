#!/bin/bash
# Bolt entry for kinova_sweep.py: build the dataset cache from the portable .npz, restore/save ../log checkpoints.
set -uo pipefail
cd "$(dirname "$0")/.."
ART="${ARTIFACT_DIR:-/mnt/task_wrapper/user_output/artifacts}"; mkdir -p "$ART" ../log
PY=""; for P in /workspace/isaaclab/_isaac_sim/python.sh python3; do command -v "$P" >/dev/null 2>&1 || [ -x "$P" ] || continue; "$P" -c "import torch, pandas" >/dev/null 2>&1 && { PY="$P"; break; }; done
export PY
for z in ../data/datasets/*.cache.npz; do
  [ -f "$z" ] || continue; pt="${z%.cache.npz}.pt"; [ -f "$pt" ] && continue
  "$PY" - "$z" "$pt" <<'PYEOF'
import sys, numpy as np, torch
z = np.load(sys.argv[1]); ns = {k[3:]: z[k] for k in z.files if k.startswith("ns_")}
torch.save({"Ktrain_data": z["Ktrain_data"], "Kval_data": z["Kval_data"], "Ktest_data": z["Ktest_data"],
            "norm_stats": ns or None}, sys.argv[2]); print("[kinova] built", sys.argv[2])
PYEOF
done
CK="${BOLT_CHECKPOINT_DIR:-}"
if [ -n "$CK" ] && [ -f "$CK/log.tar" ]; then echo "[kinova] restoring log/"; tar -xf "$CK/log.tar" -C ..; fi
if [ -n "$CK" ]; then
  "$PY" -c "import apple_bolt" 2>/dev/null || "$PY" -m pip install -q apple-bolt --index-url https://pypi.apple.com/simple >/dev/null 2>&1
  ( while sleep 901; do tar -cf "$CK/log.tar.tmp" -C .. log 2>/dev/null && mv "$CK/log.tar.tmp" "$CK/log.tar" \
      && "$PY" -c "import apple_bolt; apple_bolt.checkpoint()" >/dev/null 2>&1; done ) > /dev/null 2>&1 &
  CKPT_PID=$!
fi
trap '[ -n "${CKPT_PID:-}" ] && kill $CKPT_PID 2>/dev/null; mkdir -p "$ART/logs"; for d in ../log/Kinova_v3__*; do [ -d "$d" ] && cp "$d/koopman_results_log.csv" "$ART/logs/$(basename $d).csv" 2>/dev/null; done' EXIT
"$PY" kinova_sweep.py
