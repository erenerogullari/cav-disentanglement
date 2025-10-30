import os
import hydra
import torch
import logging
import numpy as np
from omegaconf import DictConfig, OmegaConf
from experiments.diffae import run_encode, run_move_encs, train_dir_model

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../configs", config_name="run_diffae")
def run(cfg: DictConfig) -> None:
    """Main function to run the disentangle_cavs experiment.
    Args:
        cfg (DictConfig): Configuration object containing all parameters.
    Returns:
        None  
    """
    device = cfg.experiment.device
    log.info(f"Using device: {device}")

    log.info("1. Encoding activations:")
    encodings, labels = run_encode(cfg)

    log.info("2. Training direction model:")
    dir_models = train_dir_model(cfg, encodings, labels)

    # log.info("3. Moving encodings to desired label:")
    # moved_encodings = run_move_encs(cfg, encodings, labels, dir_models)

    # log.info("4. Decoding moved encodings to images:")
    


    log.info("Experiment succesfully completed.")

if __name__ == "__main__":
    run()