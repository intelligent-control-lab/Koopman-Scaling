#!/bin/bash

# =================================================================
# == Hyperparameter Experiment Runner Script ==
# =================================================================
# Scaling-law experiment:
#   m = coefficient * n_eff * ln(n_eff)
#   n_eff = encode_dim_multiplier * state_dim(env)
#
# Envs: G1, Go2
# encode_dim = multiplier, NOT raw dimension
# coefficients = (1, 10, 100)
# control + covariance loss = True
# =================================================================

set -euo pipefail

# -----------------------------
# Experiment configuration
# -----------------------------
PROJECT_NAME="Sep_21_m_nlogn_corrected"

ENVS=("G1" "Go2")
SEEDS=(17382 76849 20965 84902 51194)

# encode_dim is a MULTIPLIER
ENCODE_DIMS=(1 2 4 8 16)

# coefficients for m = c * n * ln(n)
COEFFS=(5 20 40)

LAYER_DEPTHS=(3)
HIDDEN_DIMS=(256)

RESIDUALS=(True)
CONTROL_LOSSES=(True)
COVARIANCE_LOSSES=(True)

MS=(0)
MULT_BY_INPUT=(True)

# -----------------------------
# Logging
# -----------------------------
LOG_DIR="../log/${PROJECT_NAME}"
LOG_FILE="${LOG_DIR}/koopman_results_log.csv"

mkdir -p "$LOG_DIR"
touch "$LOG_FILE"

# -----------------------------
# Environment → state_dim map
# -----------------------------
get_state_dim() {
  case "$1" in
    G1)  echo 53 ;;
    Go2) echo 35 ;;
    *)
      echo "Unknown env: $1" >&2
      exit 1
      ;;
  esac
}

# -----------------------------
# Compute sample size
# m = coeff * n_eff * ln(n_eff)
# -----------------------------
compute_sample_size() {
  local env="$1"
  local mult="$2"
  local coeff="$3"

  local state_dim
  state_dim="$(get_state_dim "$env")"

  python - <<PY
import math
state_dim = ${state_dim}
mult = ${mult}
coeff = ${coeff}

n_eff = state_dim * mult
m = coeff * n_eff * math.log(n_eff)
print(max(int(round(m)), 1))
PY
}

# -----------------------------
# Count total jobs
# -----------------------------
TOTAL_JOBS=0
for SEED in "${SEEDS[@]}"; do
  for ENV in "${ENVS[@]}"; do
    for ENCODE_DIM in "${ENCODE_DIMS[@]}"; do
      for COEFF in "${COEFFS[@]}"; do
        for LAYER_DEPTH in "${LAYER_DEPTHS[@]}"; do
          for HIDDEN_DIM in "${HIDDEN_DIMS[@]}"; do
            for RESIDUAL in "${RESIDUALS[@]}"; do
              for CONTROL_LOSS in "${CONTROL_LOSSES[@]}"; do
                for COVARIANCE_LOSS in "${COVARIANCE_LOSSES[@]}"; do
                  for M in "${MS[@]}"; do
                    for MULT in "${MULT_BY_INPUT[@]}"; do
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

echo "✅ Total jobs to check: $TOTAL_JOBS"

# -----------------------------
# Main loop
# -----------------------------
RUN_COUNT=0
SKIP_COUNT=0
CURRENT_JOB=0

for SEED in "${SEEDS[@]}"; do
  for ENV in "${ENVS[@]}"; do
    for ENCODE_DIM in "${ENCODE_DIMS[@]}"; do
      for COEFF in "${COEFFS[@]}"; do

        SAMPLE_SIZE="$(compute_sample_size "$ENV" "$ENCODE_DIM" "$COEFF")"

        for LAYER_DEPTH in "${LAYER_DEPTHS[@]}"; do
          for HIDDEN_DIM in "${HIDDEN_DIMS[@]}"; do
            for RESIDUAL in "${RESIDUALS[@]}"; do
              for CONTROL_LOSS in "${CONTROL_LOSSES[@]}"; do
                for COVARIANCE_LOSS in "${COVARIANCE_LOSSES[@]}"; do
                  for M in "${MS[@]}"; do
                    for MULT in "${MULT_BY_INPUT[@]}"; do

                      CURRENT_JOB=$((CURRENT_JOB + 1))

                      RESIDUAL_STR="True"
                      CONTROL_LOSS_STR="True"
                      COVARIANCE_LOSS_STR="True"
                      MULT_STR="times_input_dim"

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
                         NR>1 {
                           sub(/\r/,"", $37)
                           if ($1 == env && $5 == sample_size && $27 == seed &&
                               $22 == encode_dim && $25 == layer_depth &&
                               $17 == residual && $18 == control_loss &&
                               $19 == covariance_loss && $24 == mult_mode &&
                               $37 == m_val) {
                             found=1; exit
                           }
                         }
                         END { print found }' "$LOG_FILE")

                      if [ "$IS_COMPLETED" -eq 1 ]; then
                        echo "⏭️ [$CURRENT_JOB/$TOTAL_JOBS] Skip: ENV=$ENV EDIM×=$ENCODE_DIM COEFF=$COEFF SSIZE=$SAMPLE_SIZE"
                        SKIP_COUNT=$((SKIP_COUNT + 1))
                        continue
                      fi

                      CMD="python train_model.py \
                        --env_name $ENV \
                        --seed $SEED \
                        --encode_dim $ENCODE_DIM \
                        --sample_size $SAMPLE_SIZE \
                        --layer_depth $LAYER_DEPTH \
                        --hidden_dim $HIDDEN_DIM \
                        --m $M \
                        --use_residual \
                        --use_control_loss \
                        --use_covariance_loss \
                        --multiply_encode_by_input_dim"

                      echo "----------------------------------------------------"
                      echo "🚀 RUN $CURRENT_JOB / $TOTAL_JOBS"
                      echo "ENV=$ENV  state_dim=$(get_state_dim "$ENV")"
                      echo "encode_mult=$ENCODE_DIM  n_eff=$((ENCODE_DIM * $(get_state_dim "$ENV")))"
                      echo "coeff=$COEFF  sample_size=$SAMPLE_SIZE"
                      echo "CMD: $CMD"
                      echo "----------------------------------------------------"

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
echo "🎉 Sweep finished"
echo "Total checked: $TOTAL_JOBS"
echo "Run: $RUN_COUNT"
echo "Skipped: $SKIP_COUNT"
echo "===================================================="
