#!/usr/bin/env bash
set -euo pipefail

# ------------- Hyperparameters  -------------
HARDWARE="local"          # Options: local, workstation
MODEL="vit_b_32"                # Options: vgg16, resnet18, simplenet, lenet5, vit_b_32, vit_b_16
DATASET="celeba"                # Options: celeba, elements_standart
LAYER="inspection_layer"

CAV_MODEL="pattern_cav"          # Options: pattern_cav, multi_cav
CAV_MODE="full"
OPTIMAL_INIT="false"             # true = CAV finetuning, false = training from scratch
EXIT_CRITERION="None"            # Options: None, orthogonality, auc
NUM_EPOCHS="10"
LRS=("0.0001")                   # Learning rate for CAV optimization
ALPHAS=("1")      # Regularization weights for orthogonalization

# ---------------------------------------------

for ALPHA in "${ALPHAS[@]}"; do
  for LR in "${LRS[@]}"; do
    echo "Running disentanglement with alpha=${ALPHA} and learning_rate=${LR} on layer=${LAYER}"
    python -m experiments.run_cav_disentanglement \
      hardware@train="${HARDWARE}" \
      model="${MODEL}" \
      dataset="${DATASET}" \
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
done