import os
import hydra
import torch
import logging
import numpy as np
from omegaconf import DictConfig, OmegaConf
from experiments.utils.train_cavs import train_cavs
from experiments.clarc.evaluate_model_correction import evaluate_model_correction

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
    evaluate_model_correction(cfg, cav_model)   # TODO

    log.info("3. Evaluating heatmaps.")
    evaluate_model_correction(cfg, cav_model)   # TODO

    pass

if __name__ == "__main__":
    run()