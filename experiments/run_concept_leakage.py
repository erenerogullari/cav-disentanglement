import logging

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from experiments.concept_leakage import evaluate_concept_leakage
from experiments.utils.train_cavs import (
    _load_model,
    _resolve_checkpoint_path,
    train_cavs,
)
from experiments.utils.utils import get_save_dir

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="concept_leakage")
def run(cfg: DictConfig) -> None:
    save_dir = get_save_dir(cfg)
    cav_path = save_dir / "cavs.pt"
    state_path = save_dir / "state_dict.pth"

    if cav_path.is_file() and state_path.is_file():
        log.info("Using cached final-epoch CAVs from %s.", save_dir)
    else:
        log.info("1. Training final-epoch CAVs.")
        train_cavs(cfg)

    save_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, save_dir / "config.yaml", resolve=True)

    log.info("2. Loading COCO validation data and SSD-VGG model.")
    validation_cfg = OmegaConf.create(
        OmegaConf.to_container(cfg.dataset, resolve=True)
    )
    validation_cfg.split = cfg.evaluation.split
    validation_cfg.max_samples = None
    validation_dataset = instantiate(validation_cfg)

    checkpoint_path = _resolve_checkpoint_path(cfg.model, cfg.dataset.name)
    model = _load_model(cfg.model, checkpoint_path, cfg.evaluation.device)
    cavs = torch.load(cav_path, map_location="cpu", weights_only=True)

    log.info("3. Evaluating Dining-6 localization and directed leakage.")
    evaluate_concept_leakage(
        cfg,
        model,
        validation_dataset,
        cavs,
        save_dir,
    )
    log.info("Concept-leakage experiment completed: %s", save_dir)


if __name__ == "__main__":
    run()
