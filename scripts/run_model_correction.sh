#!/usr/bin/env bash
set -euo pipefail

# Hyper parameters
ALPHAS=("0.01" "0.1" "1" "10" "100")

for ALPHA in "${ALPHAS[@]}"; do
  echo "Running model correction experiment with alpha=${ALPHA}"
  python -m experiments.run_model_correction \
    cav.alpha="${ALPHA}" \
    "$@"
done