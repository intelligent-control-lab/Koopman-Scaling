#!/usr/bin/env bash
CSV=../log/Sep_13/koopman_results_log.csv
SAVE_PATH=../log/Sep_13/isaac_control_results.csv
TOTAL_MODELS=800

while true; do
  if [ -f "$SAVE_PATH" ]; then processed=$(( $(wc -l < "$SAVE_PATH") - 1 )); else processed=0; fi
  echo "➡️  $processed / $TOTAL_MODELS models processed so far."
  if [ "$processed" -ge "$TOTAL_MODELS" ]; then echo "✅ All $TOTAL_MODELS models processed. Exiting."; break; fi

  echo "▶️  Running from start_idx=$processed …"
  # bail out if python returns non-zero
  python mpc_tracking.py --headless --start_idx "$processed" \
    --csv_log_path "$CSV" --save_path "$SAVE_PATH" || exit $?

  echo "✅ Batch complete. Looping…"
  echo
done