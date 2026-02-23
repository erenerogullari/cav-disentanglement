#!/usr/bin/env bash
set -euo pipefail

# ------------- Hyperparameters  -------------
HARDWARE="workstation"          # Options: local, workstation
MODEL="vgg16"               # Options: vgg16, resnet18, simplenet, lenet5
DATASET="celeba"     # Options: celeba, elements_standart
LAYER="features.28"
# LAYERS=("features.0" "features.2" "features.5" "features.7" "features.10" "features.12" "features.14" "features.17" "features.19" "features.21" "features.24" "features.26")

CAV_MODEL="multi_cav"         # Options: pattern_cav, multi_cav
OPTIMAL_INIT="false"             # true = CAV finetuning, false = training from scratch
EXIT_CRITERION="None"            # Options: None, orthogonality, auc
NUM_EPOCHS="100"
LRS=("0.0001")                   # Learning rate for CAV optimization
ALPHAS=("1000" "10000")      # Regularization weights for orthogonalization

# ---------------------------------------------

for ALPHA in "${ALPHAS[@]}"; do
  for LR in "${LRS[@]}"; do
    echo "Running disentanglement with alpha=${ALPHA} and learning_rate=${LR} on layer=${LAYER}"
    python -m experiments.run_cav_disentanglement \
      hardware@train="${HARDWARE}" \
      model="${MODEL}" \
      dataset="${DATASET}" \
      cav_model@cav="${CAV_MODEL}" \
      cav.optimal_init="${OPTIMAL_INIT}" \
      cav.exit_criterion="${EXIT_CRITERION}" \
      cav.layer="${LAYER}" \
      train.num_epochs="${NUM_EPOCHS}" \
      train.learning_rate="${LR}" \
      cav.alpha="${ALPHA}" \
    "$@"
  done
done