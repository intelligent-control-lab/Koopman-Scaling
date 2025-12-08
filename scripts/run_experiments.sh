#!/bin/bash

# =================================================================
# == Hyperparameter Experiment Runner Script ==
# =================================================================
# This script iterates through all hyperparameter combinations and
# runs the train_model.py script for each one.
# It checks a log file to avoid re-running completed experiments.
# =================================================================

# --- Hyperparameter Definitions ---
PROJECT_NAME="Sep_21"
ENVS=("Polynomial") #("DampingPendulum" "DoublePendulum" "Franka" "Kinova" "G1" "Go2", "Polynomial")
SEEDS=(17382 76849 20965 84902 51194)
ENCODE_DIMS=(1 2 4 8 16)
SAMPLE_SIZES=(1000 4000 16000 64000 140000)
LAYER_DEPTHS=(3)
HIDDEN_DIMS=(256)
RESIDUALS=(True)
CONTROL_LOSSES=(False) #(False True)
COVARIANCE_LOSSES=(False True)
MS=(400)
MULT_BY_INPUT=(True) # Corresponds to multiply_encode_by_input_dim

# --- Log File Configuration ---
LOG_DIR="../log/${PROJECT_NAME}"
LOG_FILE="${LOG_DIR}/koopman_results_log.csv"

# --- Script Logic ---
# Ensure log directory exists
mkdir -p "$LOG_DIR"
# Touch the file to ensure it exists for the awk command
touch "$LOG_FILE"

# --- Calculate Total Jobs ---
TOTAL_JOBS=0
for SEED in "${SEEDS[@]}"; do
  for ENV in "${ENVS[@]}"; do
    for ENCODE_DIM in "${ENCODE_DIMS[@]}"; do
      for SAMPLE_SIZE in "${SAMPLE_SIZES[@]}"; do
        for LAYER_DEPTH in "${LAYER_DEPTHS[@]}"; do
          for HIDDEN_DIM in "${HIDDEN_DIMS[@]}"; do
            for RESIDUAL in "${RESIDUALS[@]}"; do
              for CONTROL_LOSS in "${CONTROL_LOSSES[@]}"; do
                for COVARIANCE_LOSS in "${COVARIANCE_LOSSES[@]}"; do
                  for M in "${MS[@]}"; do
                    for MULT in "${MULT_BY_INPUT[@]}"; do
                      # Skip invalid combinations from original script logic
                      if [[ ("$ENV" == "Polynomial" || "$ENV" == "LogisticMap") && "$CONTROL_LOSS" == "True" ]]; then
                        continue
                      fi
                      TOTAL_JOBS=$((TOTAL_JOBS + 1))
                    done
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done

echo "✅ Total number of hyperparameter combinations to check: $TOTAL_JOBS"

# --- Main Loop ---
RUN_COUNT=0
SKIP_COUNT=0
CURRENT_JOB=0

for SEED in "${SEEDS[@]}"; do
  for ENV in "${ENVS[@]}"; do
    for ENCODE_DIM in "${ENCODE_DIMS[@]}"; do
      for SAMPLE_SIZE in "${SAMPLE_SIZES[@]}"; do
        for LAYER_DEPTH in "${LAYER_DEPTHS[@]}"; do
          for HIDDEN_DIM in "${HIDDEN_DIMS[@]}"; do
            for RESIDUAL in "${RESIDUALS[@]}"; do
              for CONTROL_LOSS in "${CONTROL_LOSSES[@]}"; do
                for COVARIANCE_LOSS in "${COVARIANCE_LOSSES[@]}"; do
                  for M in "${MS[@]}"; do
                    for MULT in "${MULT_BY_INPUT[@]}"; do

                      # Skip invalid combinations
                      if [[ ("$ENV" == "Polynomial" || "$ENV" == "LogisticMap") && "$CONTROL_LOSS" == "True" ]]; then
                        continue
                      fi
                      CURRENT_JOB=$((CURRENT_JOB + 1))

                      # Convert shell boolean to the string format logged by the Python script
                      RESIDUAL_STR=$( [[ "$RESIDUAL" == "True" ]] && echo "True" || echo "False" )
                      CONTROL_LOSS_STR=$( [[ "$CONTROL_LOSS" == "True" ]] && echo "True" || echo "False" )
                      COVARIANCE_LOSS_STR=$( [[ "$COVARIANCE_LOSS" == "True" ]] && echo "True" || echo "False" )
                      MULT_STR=$( [[ "$MULT" == "True" ]] && echo "times_input_dim" || echo "absolute" )

                      # Check if this combination already exists in the log file using awk
                      # This awk command checks the specific columns that identify a unique run.
                      # Columns: 1=env, 5=samples, 17=residual, 18=ctrl_loss, 19=cov_loss,
                      #          22=edim_param, 24=edim_mode, 25=layers, 27=seed
                       IS_COMPLETED=$(awk -F, -v env="$ENV" \
                                           -v seed="$SEED" \
                                           -v sample_size="$SAMPLE_SIZE" \
                                           -v encode_dim="$ENCODE_DIM" \
                                           -v layer_depth="$LAYER_DEPTH" \
                                           -v residual="$RESIDUAL_STR" \
                                           -v control_loss="$CONTROL_LOSS_STR" \
                                           -v covariance_loss="$COVARIANCE_LOSS_STR" \
                                           -v mult_mode="$MULT_STR" \
                                           -v m_val="$M" \
                                           'BEGIN { found=0 }
                                            NR>1 { # Skip header row
                                              # FIX: Remove potential carriage return from the last field
                                              sub(/\r/,"", $37)
                                              if ($1 == env && $5 == sample_size && $27 == seed && $22 == encode_dim && $25 == layer_depth && $17 == residual && $18 == control_loss && $19 == covariance_loss && $24 == mult_mode && $37 == m_val) {
                                                found=1; exit
                                              }
                                            }
                                            END { print found }' "$LOG_FILE")

                      if [ "$IS_COMPLETED" -eq 1 ]; then
                          echo "⏭️ [${CURRENT_JOB}/${TOTAL_JOBS}] Skipping completed run: ENV=$ENV, SEED=$SEED, EDIM=$ENCODE_DIM, SSIZE=$SAMPLE_SIZE, M=$M, CTRL=$CONTROL_LOSS, COV=$COVARIANCE_LOSS"
                          SKIP_COUNT=$((SKIP_COUNT + 1))
                          continue
                      fi

                      # Build the python command
                      CMD="python train_model.py --env_name $ENV --seed $SEED --encode_dim $ENCODE_DIM --sample_size $SAMPLE_SIZE --layer_depth $LAYER_DEPTH --hidden_dim $HIDDEN_DIM --m $M"

                      # Add boolean flags only if they are true
                      if [ "$RESIDUAL" == "True" ]; then CMD="$CMD --use_residual"; fi
                      if [ "$CONTROL_LOSS" == "True" ]; then CMD="$CMD --use_control_loss"; fi
                      if [ "$COVARIANCE_LOSS" == "True" ]; then CMD="$CMD --use_covariance_loss"; fi
                      if [ "$MULT" == "True" ]; then CMD="$CMD --multiply_encode_by_input_dim"; fi

                      echo "----------------------------------------------------"
                      echo "🚀 RUNNING JOB ${CURRENT_JOB} / ${TOTAL_JOBS}"
                      echo "   COMMAND: $CMD"
                      echo "----------------------------------------------------"

                      # Execute the command
                      eval $CMD
                      RUN_COUNT=$((RUN_COUNT + 1))

                    done
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done

echo "===================================================="
echo "🎉 Hyperparameter sweep finished. 🎉"
echo "Total combinations checked: $TOTAL_JOBS"
echo "Jobs run in this session: $RUN_COUNT"
echo "Jobs skipped (already complete): $SKIP_COUNT"
echo "===================================================="