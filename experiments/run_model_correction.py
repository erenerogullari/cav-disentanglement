import os
import hydra
import torch
import logging
import numpy as np
from omegaconf import DictConfig, OmegaConf
from experiments.model_correction import evaluate_model_correction, evaluate_concept_heatmaps, get_dir_models

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../configs", config_name="model_correction")
def run(cfg: DictConfig) -> None:
    """Main function to run the concept supression experiment.
    Args:
        cfg (DictConfig): Configuration object containing all parameters.
    Returns:
        None  
    """
    device = cfg.train.device
    log.info(f"Using device: {device}")

    log.info("1. Computing CAVs.")
    dir_model, base_model = get_dir_models(cfg)

    log.info("2. Evaluating model correction.")
    evaluate_model_correction(cfg, dir_model, base_model)

    log.info("3. Evaluating heatmaps.")
    evaluate_concept_heatmaps(cfg, dir_model, base_model)

    log.info("Experiment succesfully completed.")

if __name__ == "__main__":
    run()