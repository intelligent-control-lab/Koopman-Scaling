#!/usr/bin/env bash

# Path to your output CSV
SAVE_PATH="Jun_11_control_isaaclab_metrics.csv"
# How many models you expect
TOTAL_MODELS=200

while true; do
  # Count how many results we already have
  if [ -f "$SAVE_PATH" ]; then
    # subtract 1 for header
    processed=$(( $(wc -l < "$SAVE_PATH") - 1 ))
  else
    processed=0
  fi

  echo "➡️  $processed / $TOTAL_MODELS models processed so far."

  # If we've hit the target, exit
  if [ "$processed" -ge "$TOTAL_MODELS" ]; then
    echo "✅ All $TOTAL_MODELS models processed. Exiting."
    break
  fi

  # Otherwise, run the next batch
  echo "▶️  Running from start_idx=$processed …"
  python my_mpc_tracking.py --headless --start_idx "$processed"
  exit_code=$?

  if [ $exit_code -ne 0 ]; then
    echo "❌ Script failed at start_idx=$processed with exit code $exit_code"
    exit $exit_code
  fi

  echo "✅ Batch complete. Looping…"
  echo
done
