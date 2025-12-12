import torch
import torch.nn as nn
from torchvision.utils import save_image
import logging
from omegaconf import DictConfig, OmegaConf
from experiments.utils.train_cavs import train_cavs
from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple
from hydra.utils import get_original_cwd, instantiate
from experiments.utils.utils import get_save_dir
from experiments.utils.activations import extract_latents
from experiments.model_correction.utils import load_base_model
from utils.cav import compute_cavs

log = logging.getLogger(__name__)

def load_dir_model(target: str, state_path: Path) -> torch.nn.Module:
    state_dict = torch.load(state_path, map_location="cpu")
    n_concepts, n_features = state_dict["weights"].shape
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

def load_base_cav_model(cfg: DictConfig, activations: torch.Tensor, labels: torch.Tensor) -> nn.Module:
    cavs, bias = compute_cavs(activations, labels, type=cfg.cav.name, normalize=True)
    dir_model = instantiate(
        {"_target_": cfg.cav._target_},
        n_concepts=cavs.shape[0],
        n_features=cavs.shape[1],
        device=cfg.train.device,
    )
    dir_model.set_params(cavs, bias)
    dir_model.eval()
    return dir_model

def run_preprocessing(config: DictConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    log.info("Running preprocessing to extract activations and labels.")
    device = torch.device(config.train.device)
    dataset = instantiate(config.dataset)
    num_classes = len(dataset.classes)
    model = load_base_model(config, num_classes, device)
    activations, labels = extract_latents(config, model, dataset)
    return activations, labels


def get_dir_models(cfg: DictConfig) -> Tuple[nn.Module, nn.Module]:
    alpha = cfg.cav.alpha
    save_dir = get_save_dir(cfg)
    state_path = save_dir / "state_dict.pth"

    activations, labels = run_preprocessing(cfg)
    if state_path.exists():
        log.info("Found cached direction model for alpha=%s in %s.", alpha, state_path)
        dir_model = load_dir_model(cfg.cav._target_, state_path)
    else:
        log.info("No cached direction model found for alpha=%s. Training new model.", alpha)
        dir_model = train_cavs(cfg, activations, labels, save_dir)    # type: ignore

    base_model = load_base_cav_model(cfg, activations, labels)

    return dir_model, base_model
