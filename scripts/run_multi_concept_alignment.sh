#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export CELEBA_DATA_ROOT="/media/erogullari/datasets/"
export CLEAN_VGG16_CHECKPOINT="/media/erogullari/checkpoints/checkpoint_vgg16_celeba.pth"

CONCEPT_COUNTS=(1 2 3 4 5)
ALPHAS=(0.01 0.1 1 10)
CAV_NAMES=(pattern_cav multi_cav log_cav svm_cav ridge_cav random_cav)
CAV_TARGETS=(
  cav_models.PatternCAV
  cav_models.MultiPatternCAV
  cav_models.LogCAV
  cav_models.SvmCAV
  cav_models.RidgeCAV
  cav_models.RandomCAV
)

for K in "${CONCEPT_COUNTS[@]}"; do
  for INDEX in "${!CAV_NAMES[@]}"; do
    CAV_NAME="${CAV_NAMES[$INDEX]}"
    CAV_TARGET="${CAV_TARGETS[$INDEX]}"
    for ALPHA in "${ALPHAS[@]}"; do
      echo "Running K=${K}, CAV=${CAV_NAME}, alpha=${ALPHA}"
      python3 -m experiments.run_multi_concept_alignment \
        dataset.num_concepts="${K}" \
        cav._target_="${CAV_TARGET}" \
        cav.name="${CAV_NAME}" \
        cav.alpha="${ALPHA}" \
        "$@"
    done
  done
done
