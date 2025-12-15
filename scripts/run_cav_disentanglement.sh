#!/usr/bin/env bash
set -euo pipefail

# ------------- Hyperparameters  -------------
HARDWARE="workstation"          # Options: local, workstation
MODEL="simplenet"               # Options: vgg16, resnet18, simplenet, lenet5
DATASET="elements_standart"     # Options: celeba, elements_standart
LAYER="features.30"

CAV_MODEL="pattern_cav"         # Options: pattern_cav, multi_cav
OPTIMAL_INIT="false"            # true = CAV finetuning, false = training from scratch
EXIT_CRITERION="auc"            # Options: None, orthogonality, auc
NUM_EPOCHS="200"
LR="0.001"                    # Learning rate for CAV optimization
ALPHAS=("0" "0.01" "0.1" "1" "10" "100")      # Regularization weights for orthogonalization

# ---------------------------------------------

for ALPHA in "${ALPHAS[@]}"; do
  echo "Running disentanglement with alpha=${ALPHA}"
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
