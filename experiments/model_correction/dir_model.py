import torch
import torch.nn as nn
import logging
from omegaconf import DictConfig, OmegaConf
from experiments.utils.train_cavs import train_cavs
from pathlib import Path
from hydra.utils import get_original_cwd, instantiate

log = logging.getLogger(__name__)

def load_dir_model(target: str, n_concepts: int, n_features: int, state_path: Path) -> torch.nn.Module:
    dir_model = instantiate(
        {"_target_": target},
        n_concepts=n_concepts,
        n_features=n_features,
        device="cpu",
    )
    state_dict = torch.load(state_path, map_location="cpu")
    dir_model.load_state_dict(state_dict)
    dir_model.eval()
    return dir_model

def prepare_config(cfg: DictConfig, alpha: float) -> DictConfig:
    cav_cfg = OmegaConf.create(
        {
            "experiment": {"name": cfg.experiment.name},
            "dataset": cfg.encode.dataset,
            "model": cfg.model,
            "train": {
                "learning_rate": cfg.dir_model.learning_rate,
                "num_epochs": cfg.dir_model.n_epochs,
                "batch_size": cfg.experiment.batch_size,
                "num_workers": cfg.experiment.num_workers,
                "device": cfg.experiment.device,
                "random_seed": cfg.dir_model.random_seed,
                "val_ratio": cfg.dir_model.val_ratio,
                "test_ratio": cfg.dir_model.test_ratio,
            },
            "cav": {
                "_target_": cfg.dir_model["_target_"],
                "name": cfg.dir_model.name,
                "layer": "",
                "alpha": alpha,
                "beta": None,
                "n_targets": 0,
                "optimal_init": False,
                "exit_criterion": "auc",
                "cav_mode": getattr(cfg.move_encs, "cav_mode", "max"),
            },
        }
    )

    return DictConfig(cav_cfg)

def get_dir_model(cfg: DictConfig) -> nn.Module:
    cache_dir = Path("results") / "clarc" / str(cfg.encode.dataset.name) / "dir_models" / str(cfg.dir_model.name)
    alpha = cfg.dir_model.alpha
    save_dir = cache_dir / f"alpha{alpha}"
    state_path = save_dir / "state_dict.pth"

    if state_path.exists():
        log.info("Found cached direction model for alpha=%s in %s.", alpha, state_path)
        dir_model = load_dir_model(cfg.dir_model["_target_"], state_path)
        return dir_model

    log.info("No cached direction model found for alpha=%s. Training new model.", alpha)
    cav_cfg = prepare_config(cfg, alpha)
    labels_clamped = labels.clamp(0)
    dir_model = train_cavs(cav_cfg, encodings, labels_clamped, save_dir)    # type: ignore

    return dir_model
