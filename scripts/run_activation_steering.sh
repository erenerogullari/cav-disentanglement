#!/usr/bin/env bash
set -euo pipefail

# ------------- Hyperparameters  -------------
ALPHAS=("1" "10" "100")      # Regularization weights for orthogonalization
# CAV_TRAIN_RATIO="${CAV_TRAIN_RATIO:-}"      # Optional ratio in (0, 1] for CAV training rows only
# CAV_TRAIN_SUBSET_SEED="${CAV_TRAIN_SUBSET_SEED:-}"
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

echo "Running activation steering experiment with alpha=0"
python -m experiments.run_activation_steering \
  dir_model.alpha="0" \
  dir_model.n_epochs="10" \
  dir_model.exit_criterion="auc" \
  "${SUBSET_OVERRIDES[@]}" \
  "$@"

for ALPHA in "${ALPHAS[@]}"; do
  echo "Running activation steering experiment with alpha=${ALPHA}"
  python -m experiments.run_activation_steering \
    dir_model.alpha="${ALPHA}" \
    "${SUBSET_OVERRIDES[@]}" \
    "$@"
done
