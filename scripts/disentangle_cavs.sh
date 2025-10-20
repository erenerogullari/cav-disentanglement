#!/usr/bin/env bash
set -euo pipefail

# ------------- Hyperparameters  -------------
HARDWARE="local"                # Options: local, workstation
MODEL="vgg16"                   # Options: vgg16, resnet18
DATASET="celeba"                # Options: celeba, elements
LAYER="features.28"

CAV_MODEL="pattern_cav"          # Options: pattern_cav, multi_cav
OPTIMAL_INIT="false"            # true = CAV finetuning, false = training from scratch
NUM_EPOCHS="200"
LR="0.01"                     # Learning rate for CAV optimization
ALPHA="0"                       # Regularization weight for orthogonalization

# ---------------------------------------------

exec python -m experiments.disentangle_cavs\
  hardware@train="${HARDWARE}" \
  model="${MODEL}" \
  dataset="${DATASET}" \
  cav_model@cav="${CAV_MODEL}" \
  cav.optimal_init="${OPTIMAL_INIT}" \
  cav.layer="${LAYER}" \
  train.num_epochs="${NUM_EPOCHS}" \
  train.learning_rate="${LR}" \
  cav.alpha="${ALPHA}" \
  "$@"
