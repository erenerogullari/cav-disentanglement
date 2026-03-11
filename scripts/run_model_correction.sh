#!/usr/bin/env bash
set -euo pipefail

# Hyper parameters
HARDWARE="workstation"          # Options: local, workstation
MODEL="vit_b_32"           # Options: vit_b_32, vit_b_16, vgg16
LAYER="inspection_layer"     

CAV_MODEL="pattern_cav"         # Options: pattern_cav, multi_cav
OPTIMAL_INIT="true"            # true = CAV finetuning, false = training from scratch
EXIT_CRITERION="orthogonality"            # Options: None, orthogonality, auc
NUM_EPOCHS="5"
LR="0.0001"                    # Learning rate for CAV optimization
# ALPHAS=("0.01" "0.1" "1" "10")
ALPHAS=("1")

for ALPHA in "${ALPHAS[@]}"; do
  echo "Running model correction experiment with alpha=${ALPHA}"
  python -m experiments.run_model_correction \
    hardware@train="${HARDWARE}" \
    model="${MODEL}" \
    cav_model@cav="${CAV_MODEL}" \
    cav.optimal_init="${OPTIMAL_INIT}" \
    cav.exit_criterion="${EXIT_CRITERION}" \
    cav.layer="${LAYER}" \
    train.num_epochs="${NUM_EPOCHS}" \
    train.learning_rate="${LR}" \
    cav.alpha="${ALPHA}" \
    "$@"
done