import os
import hydra
import torch
import logging
import numpy as np
from omegaconf import DictConfig, OmegaConf
from experiments.utils.train_cavs import train_cavs
from experiments.clarc import evaluate_model_correction, evaluate_concept_heatmaps

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../configs", config_name="run_pclarc")
def run(cfg: DictConfig) -> None:
    """Main function to run the concept supression experiment.
    Args:
        cfg (DictConfig): Configuration object containing all parameters.
    Returns:
        None  
    """
    log.info(f"Starting experiment: {cfg.experiment.name}.")

    log.info("1. Training CAVs.")
    cav_model = train_cavs(cfg, None, None)

    log.info("2. Evaluating model correction.")
    evaluate_model_correction(cfg, cav_model)

    log.info("3. Evaluating heatmaps.")
    evaluate_concept_heatmaps(cfg, cav_model)

    log.info("Experiment succesfully completed.")

if __name__ == "__main__":
    run()