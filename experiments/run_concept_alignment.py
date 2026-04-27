import hydra
import logging
from omegaconf import DictConfig
from experiments.concept_alignment import evaluate_concept_alignment
from experiments.model_correction import evaluate_concept_heatmaps
from experiments.model_correction.dir_model import get_dir_models

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None, config_path="../configs", config_name="concept_alignment"
)
def run(cfg: DictConfig) -> None:
    """Main function to run concept-alignment evaluation."""

    device = cfg.train.device
    log.info(f"Using device: {device}")

    log.info("1. Computing CAVs.")
    dir_model, base_model = get_dir_models(cfg)

    log.info("2. Evaluating concept alignment.")
    # evaluate_concept_alignment(cfg, dir_model, base_model)

    log.info("3. Evaluating concept heatmaps.")
    evaluate_concept_heatmaps(cfg, dir_model, base_model)

    log.info("Experiment succesfully completed.")


if __name__ == "__main__":
    run()
