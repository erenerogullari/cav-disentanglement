#!/usr/bin/env bash
set -euo pipefail

# ------------- Hyperparameters  -------------
ALPHA="0"           # Target pair relevance weights
BETAS=("1" "10" "100")      # Non-target pair orthogonality weights
CAV_MODELS=("pattern_cav" "multi_cav" "log_cav" "svm_cav" "random_cav")  # CAV models to run activation steering with
CAV_TRAIN_RATIO="0.1"      # Optional ratio in (0, 1] for CAV training rows only
CAV_TRAIN_SUBSET_SEED="42" 

# ---------------------------------------------
SUBSET_OVERRIDES=()
if [[ -n "${CAV_TRAIN_RATIO}" ]]; then
  SUBSET_OVERRIDES+=(dir_model.train_subset.ratio="${CAV_TRAIN_RATIO}")
fi
if [[ -n "${CAV_TRAIN_SUBSET_SEED}" ]]; then
  SUBSET_OVERRIDES+=(dir_model.train_subset.seed="${CAV_TRAIN_SUBSET_SEED}")
fi

for CAV_MODEL in "${CAV_MODELS[@]}"; do
echo "Running activation steering for ${CAV_MODEL} experiment with alpha=0, beta=0"
  python -m experiments.run_activation_steering \
    cav_model@dir_model="${CAV_MODEL}" \
    dir_model.alpha="0" \
    dir_model.beta="0" \
    dir_model.n_epochs="10" \
    "${SUBSET_OVERRIDES[@]}" \
    "$@"
done

for CAV_MODEL in "${CAV_MODELS[@]}"; do
  for BETA in "${BETAS[@]}"; do
    echo "Running activation steering for ${CAV_MODEL} experiment with alpha=${ALPHA}, beta=${BETA}"
    python -m experiments.run_activation_steering \
      cav_model@dir_model="${CAV_MODEL}" \
      dir_model.alpha="${ALPHA}" \
      dir_model.beta="${BETA}" \
      "${SUBSET_OVERRIDES[@]}" \
      "$@"
  done
done
