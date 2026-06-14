import torch
import torch.nn as nn
import logging
from omegaconf import DictConfig, OmegaConf
from experiments.utils.train_cavs import train_cavs
from experiments.utils.utils import (
    format_orthogonality_config_name,
    get_target_concepts,
)
from pathlib import Path
from hydra.utils import instantiate

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

def prepare_config(cfg: DictConfig) -> DictConfig:
    alpha = cfg.dir_model.alpha
    beta = cfg.dir_model.get("beta", None)
    target_concepts = get_target_concepts(cfg.dir_model)
    train_cfg = {
        "learning_rate": cfg.dir_model.learning_rate,
        "num_epochs": cfg.dir_model.n_epochs,
        "batch_size": cfg.experiment.batch_size,
        "num_workers": cfg.experiment.num_workers,
        "device": cfg.experiment.device,
        "random_seed": cfg.dir_model.random_seed,
        "val_ratio": cfg.dir_model.val_ratio,
        "test_ratio": cfg.dir_model.test_ratio,
    }
    if "train_subset" in cfg.dir_model:
        train_cfg["subset"] = OmegaConf.to_container(
            cfg.dir_model.train_subset, resolve=True
        )

    cav_cfg = OmegaConf.create(
        {
            "experiment": {"name": cfg.experiment.name},
            "dataset": cfg.encode.dataset,
            "model": cfg.model,
            "train": train_cfg,
            "cav": {
                "_target_": cfg.dir_model["_target_"],
                "name": cfg.dir_model.name,
                "layer": cfg.dir_model.get("layer", "bottleneck"),
                "alpha": alpha,
                "beta": beta,
                "target_concepts": target_concepts,
                "optimal_init": cfg.dir_model.optimal_init,
                "exit_criterion": cfg.dir_model.exit_criterion,
                "cav_mode": getattr(cfg.move_encs, "cav_mode", "max"),
            },
        }
    )

    return DictConfig(cav_cfg)

def get_dir_model(cfg: DictConfig, encodings: torch.Tensor, labels: torch.Tensor) -> nn.Module:
    cav_cfg = prepare_config(cfg)
    cache_dir = Path(cfg.experiment.out) / "dir_models" / str(cfg.dir_model.name)
    target_concepts = get_target_concepts(cfg.dir_model)
    cache_name = format_orthogonality_config_name(
        cfg.dir_model.alpha,
        cfg.dir_model.get("beta", None),
        target_concepts,
    )
    save_dir = cache_dir / cache_name
    state_path = save_dir / "state_dict.pth"

    if state_path.exists():
        log.info("Found cached direction model for %s in %s.", cache_name, state_path)
        dir_model = load_dir_model(cfg.dir_model["_target_"], labels.shape[1], encodings.shape[1], state_path)
        return dir_model

    log.info("No cached direction model found for %s. Training new model.", cache_name)
    labels_clamped = labels.clamp(0)
    dir_model = train_cavs(cav_cfg, encodings, labels_clamped, save_dir)    # type: ignore

    return dir_model
