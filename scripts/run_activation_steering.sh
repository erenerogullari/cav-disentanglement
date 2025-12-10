#!/usr/bin/env bash
set -euo pipefail

# ------------- Hyperparameters  -------------
ALPHAS=("0.01" "0.1" "1" "10" "100")      # Regularization weights for orthogonalization

# ---------------------------------------------
echo "Running activation steering experiment with alpha=0"
python -m experiments.run_activation_steering \
  dir_model.alpha="0" \
  dir_model.n_epochs="10" \
  dir_model.exit_criterion="auc" \
  "$@"

for ALPHA in "${ALPHAS[@]}"; do
  echo "Running activation steering experiment with alpha=${ALPHA}"
  python -m experiments.run_activation_steering \
    dir_model.alpha="${ALPHA}" \
    "$@"
done
