import os
import hydra
import torch
import logging
import numpy as np
from omegaconf import DictConfig, OmegaConf
# from experiments.pclarc import 

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../configs", config_name="run_pclarc")
def run(cfg: DictConfig) -> None:
    """Main function to run the concept supression experiment.
    Args:
        cfg (DictConfig): Configuration object containing all parameters.
    Returns:
        None  
    """
    device = cfg.experiment.device
    log.info(f"Using device: {device}")

    log.info("1. Training CAVs:")

    log.info("2. Model Correction:")

    pass

if __name__ == "__main__":
    run()