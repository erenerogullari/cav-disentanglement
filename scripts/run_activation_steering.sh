#!/usr/bin/env bash
set -euo pipefail

# ------------- Hyperparameters  -------------
HARDWARE="workstation"
ALPHAS=("0" "0.1" "1" "10")  # Non-target pair orthogonality weights
BETAS=("0.1" "1" "10")        # Target-involving pair orthogonality weights
TARGET_CONCEPTS="[Wearing_Necktie]"  # Target concepts for beta orthogonalization
MOVE_TARGET_CONCEPT="Wearing_Necktie"  # Concept inserted when moving encodings
CAV_MODELS=("multi_cav")  # CAV models to run activation steering with
CAV_TRAIN_RATIO="0.1"      # Optional ratio in (0, 1] for CAV training rows only
CAV_TRAIN_SUBSET_SEED="42" 

# ---------------------------------------------
SUBSET_OVERRIDES=()
# if [[ -n "${CAV_TRAIN_RATIO}" ]]; then
#   SUBSET_OVERRIDES+=(dir_model.train_subset.ratio="${CAV_TRAIN_RATIO}")
# fi
# if [[ -n "${CAV_TRAIN_SUBSET_SEED}" ]]; then
#   SUBSET_OVERRIDES+=(dir_model.train_subset.seed="${CAV_TRAIN_SUBSET_SEED}")
# fi

for CAV_MODEL in "${CAV_MODELS[@]}"; do
  echo "Running activation steering for ${CAV_MODEL} baseline experiment with alpha=0"
  python -m experiments.run_activation_steering \
    cav_model@dir_model="${CAV_MODEL}" \
    dir_model.alpha="0" \
    dir_model.beta="null" \
    dir_model.target_concepts="[]" \
    dir_model.n_epochs="10" \
    move_encs.target_concept="${MOVE_TARGET_CONCEPT}" \
    "${SUBSET_OVERRIDES[@]}" \
    "$@"
  echo "-----------------------------------------------------------------------------"

  for ALPHA in "${ALPHAS[@]}"; do
    echo "Running activation steering for ${CAV_MODEL} experiment with alpha=${ALPHA}"
    python -m experiments.run_activation_steering \
      cav_model@dir_model="${CAV_MODEL}" \
      dir_model.alpha="${ALPHA}" \
      dir_model.beta="null" \
      dir_model.target_concepts="[]" \
      move_encs.target_concept="${MOVE_TARGET_CONCEPT}" \
      "${SUBSET_OVERRIDES[@]}" \
      "$@"
    echo "-----------------------------------------------------------------------------"
  done

  for BETA in "${BETAS[@]}"; do
    echo "Running activation steering for ${CAV_MODEL} experiment with alpha=0, beta=${BETA}"
    python -m experiments.run_activation_steering \
      cav_model@dir_model="${CAV_MODEL}" \
      dir_model.alpha="0" \
      dir_model.beta="${BETA}" \
      dir_model.target_concepts="${TARGET_CONCEPTS}" \
      move_encs.target_concept="${MOVE_TARGET_CONCEPT}" \
      "${SUBSET_OVERRIDES[@]}" \
      "$@"
    echo "-----------------------------------------------------------------------------"
  done
done
