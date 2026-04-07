import torch
import torch.nn as nn
import logging
from omegaconf import DictConfig
from experiments.utils.train_cavs import train_cavs
from pathlib import Path
from typing import Tuple
from hydra.utils import instantiate
from experiments.utils.utils import get_save_dir
from experiments.utils.activations import extract_latents
from experiments.model_correction.utils import load_base_model
from utils.cav import compute_cavs, build_cav_cache_path
from experiments.utils.cav_model_utils import (
    instantiate_cav_model,
    validate_precomputed_g_sae_cache,
)

log = logging.getLogger(__name__)


def load_dir_model(
    cfg: DictConfig,
    state_path: Path,
    n_concepts: int,
    n_features: int,
) -> torch.nn.Module:
    dir_model = instantiate_cav_model(
        cfg.cav,
        n_concepts=n_concepts,
        n_features=n_features,
        device="cpu",
    )
    state_dict = torch.load(state_path, map_location="cpu")
    if isinstance(state_dict, dict) and isinstance(state_dict.get("state_dict"), dict):
        state_dict = state_dict["state_dict"]
    if not isinstance(state_dict, dict):
        raise ValueError(
            f"Unsupported direction model checkpoint format at '{state_path}'."
        )
    dir_model.load_state_dict(state_dict)
    dir_model.eval()
    return dir_model


def load_base_cav_model(
    cfg: DictConfig, activations: torch.Tensor, labels: torch.Tensor
) -> nn.Module:
    cav_cache_path = build_cav_cache_path(
        dataset_name=cfg.dataset.name,
        model_name=cfg.model.name,
        layer_name=cfg.cav.layer,
        cav_type=cfg.cav.name,
    )
    if cfg.cav.name == "G_SAE":
        validate_precomputed_g_sae_cache(cav_cache_path)
    cavs, bias = compute_cavs(
        activations,
        labels,
        type=cfg.cav.name,
        normalize=True,
        cache_dir=cav_cache_path,
    )
    dir_model = instantiate_cav_model(
        cfg.cav,
        n_concepts=cavs.shape[0],
        n_features=cavs.shape[1],
        device=cfg.train.device,
    )
    if not hasattr(dir_model, "set_params"):
        raise AttributeError(
            f"CAV model '{cfg.cav.name}' does not expose set_params(cavs, bias)."
        )
    dir_model.set_params(cavs, bias)
    dir_model.eval()
    return dir_model


def run_preprocessing(config: DictConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    log.info("Running preprocessing to extract activations and labels.")
    device = torch.device(config.train.device)
    dataset = instantiate(config.dataset)
    num_classes = len(dataset.classes)
    log.info(f"Loading {config.model.name} at '{config.model.ckpt_path}'")
    model = load_base_model(config, num_classes, device)
    activations, labels = extract_latents(config, model, dataset)
    return activations, labels


def get_dir_models(cfg: DictConfig) -> Tuple[nn.Module, nn.Module]:
    alpha = cfg.cav.alpha
    save_dir = get_save_dir(cfg)
    state_path = save_dir / "state_dict.pth"

    activations, labels = run_preprocessing(cfg)
    n_concepts, n_features = labels.shape[1], activations.shape[1]
    if state_path.exists():
        log.info("Found cached direction model for alpha=%s in %s.", alpha, state_path)
        dir_model = load_dir_model(
            cfg=cfg,
            state_path=state_path,
            n_concepts=n_concepts,
            n_features=n_features,
        )
    else:
        log.info(
            "No cached direction model found for alpha=%s. Training new model.", alpha
        )
        dir_model = train_cavs(cfg, activations, labels, save_dir)  # type: ignore

    base_model = load_base_cav_model(cfg, activations, labels)

    return dir_model, base_model
