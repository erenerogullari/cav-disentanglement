#!/usr/bin/env bash
set -euo pipefail

# Hyper parameters
HARDWARE="workstation"          # Options: local, workstation
MODEL="vgg16"           # Options: vit_b_32, vit_b_16, vgg16
LAYER="features.29"     
CKPT_PATH="/media/erogullari/checkpoints/checkpoint_vgg16_celeba_attacked.pth"

CAV_MODEL="pattern_cav"         # Options: pattern_cav, multi_cav
CAV_MODE="max"                # Options: full, max, avg
OPTIMAL_INIT="true"             # true = CAV finetuning, false = training from scratch
EXIT_CRITERION="orthogonality"            # Options: None, orthogonality, auc
NUM_EPOCHS="50"
LR="0.0001"                    # Learning rate for CAV optimization
ALPHAS=("0.1" "1" "10" "100")
# ALPHAS=("100")

for ALPHA in "${ALPHAS[@]}"; do
  echo "Running model correction experiment with alpha=${ALPHA}"
  python -m experiments.run_model_correction \
    hardware@train="${HARDWARE}" \
    model="${MODEL}" \
    model.ckpt_path="${CKPT_PATH}" \
    cav_model@cav="${CAV_MODEL}" \
    cav.cav_mode="${CAV_MODE}" \
    cav.optimal_init="${OPTIMAL_INIT}" \
    cav.exit_criterion="${EXIT_CRITERION}" \
    cav.layer="${LAYER}" \
    train.num_epochs="${NUM_EPOCHS}" \
    train.learning_rate="${LR}" \
    cav.alpha="${ALPHA}" \
    "$@"
done