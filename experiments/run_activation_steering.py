import os
import hydra
import torch
import logging
import numpy as np
from omegaconf import DictConfig, OmegaConf
from experiments.activation_steering import run_encode, run_move_encs, run_decode, get_dir_model
from experiments.activation_steering.utils import clean_up

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="activation_steering")
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

    log.info("2. Training direction models:")
    dir_model = get_dir_model(cfg, encodings, labels)

    log.info("3. Moving encodings to desired label:")
    run_move_encs(cfg, encodings, labels, dir_model)

    log.info("4. Decoding moved encodings to images:")
    run_decode(cfg)

    log.info("5. Cleaning up temporary files")
    clean_up(cfg)
    
    log.info("Experiment succesfully completed.")

if __name__ == "__main__":
    run()