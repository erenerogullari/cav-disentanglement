#!/usr/bin/env bash
set -euo pipefail

# Hyperparameters
HARDWARE="workstation"              # Options: local, workstation

MODELS=("vgg16" "resnet18" "vit_b_16")
LAYERS=("features.29" "last_conv" "inspection_layer")

CAV_MODELS=("pattern_cav" "multi_cav" "log_cav" "svm_cav" "ridge_cav" "random_cav")  
CAV_MODE="max"                      # Options: full, max, avg
OPTIMAL_INIT="true"                 # true = CAV finetuning, false = training from scratch
EXIT_CRITERION="None"               # Options: None, orthogonality, auc
NUM_EPOCHS="50"
LR="0.0001"                         # Learning rate for CAV optimization
SPLIT="test"                        # Options: train, val, test

ALPHAS=("0.1" "1" "10" "100")
BETAS=("0.1" "1" "10" "100")
TARGET_CONCEPTS="[timestamp, box, brightness]"

for CAV_MODEL in "${CAV_MODELS[@]}"; do
  for INDEX in "${!MODELS[@]}"; do
    MODEL="${MODELS[$INDEX]}"
    LAYER="${LAYERS[$INDEX]}"
    CKPT_PATH="/media/erogullari/checkpoints/checkpoint_${MODEL}_celeba_attacked.pth"

    echo "Running ${CAV_MODEL} concept alignment for ${MODEL} (${LAYER}) with alpha=0.0"
    python -m experiments.run_concept_alignment \
      hardware@train="${HARDWARE}" \
      model="${MODEL}" \
      model.ckpt_path="${CKPT_PATH}" \
      cav_model@cav="${CAV_MODEL}" \
      cav.cav_mode="${CAV_MODE}" \
      cav.optimal_init="${OPTIMAL_INIT}" \
      cav.exit_criterion="${EXIT_CRITERION}" \
      cav.layer="${LAYER}" \
      train.num_epochs="10" \
      train.learning_rate="${LR}" \
      alignment.split="${SPLIT}" \
      cav.alpha="0.0" \
      cav.beta="null" \
      cav.target_concepts="[]" \
      "$@"
    echo "----------------------------------------------------------"

    for ALPHA in "${ALPHAS[@]}"; do
      echo "Running ${CAV_MODEL} concept alignment for ${MODEL} (${LAYER}) with alpha=${ALPHA}"
      python -m experiments.run_concept_alignment \
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
        alignment.split="${SPLIT}" \
        cav.alpha="${ALPHA}" \
        cav.beta="null" \
        cav.target_concepts="[]" \
        "$@"
      echo "----------------------------------------------------------"
    done

    for BETA in "${BETAS[@]}"; do
      echo "Running ${CAV_MODEL} concept alignment for ${MODEL} (${LAYER}) with alpha=0 and beta=${BETA}"
      python -m experiments.run_concept_alignment \
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
        alignment.split="${SPLIT}" \
        cav.alpha="0" \
        cav.beta="${BETA}" \
        cav.target_concepts="${TARGET_CONCEPTS}" \
        "$@"
      echo "----------------------------------------------------------"
    done
  done
done
