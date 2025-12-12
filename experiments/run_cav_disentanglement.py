import os
import hydra
import torch
import logging
import numpy as np
from omegaconf import DictConfig, OmegaConf
from experiments.utils.train_cavs import train_cavs
from experiments.utils.localization import localize_concepts, colocalize_concept_pairs


log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../configs", config_name="cav_disentanglement")
def run(cfg: DictConfig) -> None:
    """Main function to run the disentangle_cavs experiment.
    Args:
        cfg (DictConfig): Configuration object containing all parameters.
    Returns:
        None  
    """
    device = cfg.train.device
    log.info(f"Using device: {device}")

    # 1. Train CAVs
    log.info("1. Training CAVs:")
    # train_cavs(cfg)

    # 2. Concept Localization
    if len(cfg.localization.concept_ids) > 0:
        log.info("2. Concept Localization:")
        localize_concepts(cfg)
    else:
        log.info("2. Skipping Concept Localization...")
    
    # 3. Concept Colocalization
    if len(cfg.localization.concept_pair_ids) > 0:
        log.info("3. Concept Colocalization:")
        colocalize_concept_pairs(cfg)
    else:
        log.info("3. Skipping Concept Colocalization...")

    log.info("Experiment succesfully completed.")


if __name__ == "__main__":
    run()