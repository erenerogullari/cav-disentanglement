#!/usr/bin/env bash
set -euo pipefail

# Hyper parameters
HARDWARE="workstation"          # Options: local, workstation

CAV_MODEL="pattern_cav"         # Options: pattern_cav, multi_cav
OPTIMAL_INIT="true"            # true = CAV finetuning, false = training from scratch
EXIT_CRITERION="orthogonality"            # Options: None, orthogonality, auc
NUM_EPOCHS="200"
LR="0.001"                    # Learning rate for CAV optimization
# ALPHAS=("0.01" "0.1" "1" "10" "100")
ALPHAS=("0.01")

for ALPHA in "${ALPHAS[@]}"; do
  echo "Running model correction experiment with alpha=${ALPHA}"
  python -m experiments.run_model_correction \
    hardware@train="${HARDWARE}" \
    cav_model@cav="${CAV_MODEL}" \
    cav.optimal_init="${OPTIMAL_INIT}" \
    cav.exit_criterion="${EXIT_CRITERION}" \
    train.num_epochs="${NUM_EPOCHS}" \
    train.learning_rate="0.001" \
    cav.alpha="${ALPHA}" \
    "$@"
done

# for ALPHA in "${ALPHAS[@]}"; do
#   echo "Running model correction experiment with alpha=${ALPHA}"
#   python -m experiments.run_model_correction \
#     hardware@train="${HARDWARE}" \
#     cav_model@cav="${CAV_MODEL}" \
#     cav.optimal_init="${OPTIMAL_INIT}" \
#     cav.exit_criterion="${EXIT_CRITERION}" \
#     train.num_epochs="${NUM_EPOCHS}" \
#     train.learning_rate="0.0001" \
#     cav.alpha="${ALPHA}" \
#     "$@"
# done