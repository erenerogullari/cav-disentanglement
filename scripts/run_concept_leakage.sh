#!/usr/bin/env bash
set -euo pipefail

export COCO_ROOT=/media/erogullari/datasets/coco2017

HARDWARE="workstation"              # local or workstation
MODEL="vgg16_ssd_coco"
DATASET="coco"
LAYER="features.29"
CAV_MODE="max"

CAV_MODELS=("pattern_cav" "multi_cav" "svm_cav" "log_cav" "ridge_cav")
ALPHAS=("0" "0.1" "1" "10")
BETA="null"
TARGET_CONCEPTS="[]"
NUM_EPOCHS="200"
LEARNING_RATE="0.0001"

for CAV_MODEL in "${CAV_MODELS[@]}"; do
  for ALPHA in "${ALPHAS[@]}"; do
    echo "Running concept_leakage with cav=${CAV_MODEL}, alpha=${ALPHA}"
    python -m experiments.run_concept_leakage \
      hardware@train="${HARDWARE}" \
      model="${MODEL}" \
      dataset="${DATASET}" \
      cav_model@cav="${CAV_MODEL}" \
      cav.layer="${LAYER}" \
      cav.cav_mode="${CAV_MODE}" \
      cav.alpha="${ALPHA}" \
      cav.beta="${BETA}" \
      cav.target_concepts="${TARGET_CONCEPTS}" \
      cav.exit_criterion="null" \
      train.num_epochs="${NUM_EPOCHS}" \
      train.learning_rate="${LEARNING_RATE}" \
      "$@"
  done
done
