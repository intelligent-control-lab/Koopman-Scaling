#!/bin/bash
# Bolt entry for franka_tune.py (Franka aux-loss tuning). Restores/saves ../log via Bolt checkpoints.
set -uo pipefail
cd "$(dirname "$0")/.."
ART="${ARTIFACT_DIR:-/mnt/task_wrapper/user_output/artifacts}"; mkdir -p "$ART" ../log
PY=""; for P in /workspace/isaaclab/_isaac_sim/python.sh python3; do command -v "$P" >/dev/null 2>&1 || [ -x "$P" ] || continue; "$P" -c "import torch, pandas" >/dev/null 2>&1 && { PY="$P"; break; }; done
export PY
"$PY" -c "import pybullet" 2>/dev/null || "$PY" -m pip install -q pybullet --index-url https://pypi.apple.com/simple || "$PY" -m pip install -q pybullet
"$PY" -c "import pybullet; print('[ftune] pybullet ok')" || exit 5
CK="${BOLT_CHECKPOINT_DIR:-}"
if [ -n "$CK" ] && [ -f "$CK/log.tar" ]; then echo "[ftune] restoring log/ from checkpoint"; tar -xf "$CK/log.tar" -C ..; fi
if [ -n "$CK" ]; then
  "$PY" -c "import apple_bolt" 2>/dev/null || "$PY" -m pip install -q apple-bolt --index-url https://pypi.apple.com/simple >/dev/null 2>&1
  ( while sleep 901; do tar -cf "$CK/log.tar.tmp" -C .. log 2>/dev/null && mv "$CK/log.tar.tmp" "$CK/log.tar" \
      && "$PY" -c "import apple_bolt; apple_bolt.checkpoint()" >/dev/null 2>&1; done ) > /dev/null 2>&1 &
  CKPT_PID=$!
fi
trap '[ -n "${CKPT_PID:-}" ] && kill $CKPT_PID 2>/dev/null; mkdir -p "$ART"; cp -r ../log/franka_tuning "$ART"/ 2>/dev/null' EXIT
"$PY" franka_tune.py
