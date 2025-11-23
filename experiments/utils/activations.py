import os
import torch
import torch.nn as nn
import numpy as np
import pickle
from omegaconf import DictConfig, OmegaConf
from typing import Dict, List, Tuple, Optional
from crp.attribution import CondAttribution
from zennit.composites import EpsilonPlusFlat
from datasets import get_dataset
from models import get_fn_model_loader, get_canonizer
from hydra.utils import instantiate
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from hydra.utils import get_original_cwd
import logging

log = logging.getLogger(__name__)

def _get_features(batch, layer_name, attribution, canonizers, cav_mode, device):
    compost = EpsilonPlusFlat(canonizers=canonizers)
    batch.requires_grad = True
    dummy_cond = [{"y": 0} for _ in range(len(batch))]
    attr = attribution(batch.to(device), dummy_cond, compost, record_layer=[layer_name])
    if cav_mode == "full":
        features = attr.activations[layer_name]
    elif cav_mode == "max":
        acts = attr.activations[layer_name]
        features = acts.flatten(start_dim=2).max(2)[0]
    elif cav_mode == "avg":
        acts = attr.activations[layer_name]
        features = acts.flatten(start_dim=2).mean(2)[0]
    else:
        raise ValueError(f"Invalid cav_mode: {cav_mode}. Choose from 'full', 'max', or 'avg'.")
    return features

def get_features(batch, config, attribution):

    batch.requires_grad = True
    dummy_cond = [{"y": 0} for _ in range(len(batch))]
    record_layer=[config["layer_name"]]
    attr = attribution(batch.to(config["device"]), dummy_cond, record_layer=record_layer)
    if config["cav_mode"] == "cavs_full":
        features = attr.activations[config["layer_name"]]
    else:
        # ViT support
        acts = attr.activations[config["layer_name"]]
        acts = acts if acts.dim() > 2 else acts[..., None, None]
        acts = acts.transpose(1,3).transpose(2,3) if "swin_former" in config["model_name"] else acts
        features = acts.flatten(start_dim=2).max(2)[0]
        # features = attr.activations[config["layer_name"]].flatten(start_dim=2).max(2)[0]
    return features


def extract_latents(cfg: DictConfig, model: nn.Module, dataset: torch.utils.data.Dataset) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract latent representations from a specified layer of the model for the entire dataset.
    Args:
        cfg (DictConfig): Configuration object containing model and dataset parameters.
        model (nn.Module): The neural network model from which to extract features.
        dataset (torch.utils.data.Dataset): The dataset for which to extract features.
    Returns:
        torch.Tensor: A tensor containing the extracted latent representations.
        torch.Tensor: A tensor containing the corresponding labels.
    """
    cache_dir = Path(get_original_cwd()) / "variables"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_name = f"vars_{cfg.dataset.name}_{cfg.cav.layer}_{cfg.model.name}.pth"
    cache_path = cache_dir / cache_name

    if cache_path.exists():
        log.info(f"Loading cached latents from {cache_path}.")
        vars = torch.load(cache_path, weights_only=True)
        x_latent_all = vars["encs"]
        labels = vars["labels"]
    else:
        log.info("No cached latents found. Extracting latents...")
        dataloader = DataLoader(dataset, batch_size=cfg.train.batch_size, num_workers=cfg.train.num_workers, shuffle=False)
        attribution = CondAttribution(model)
        canonizer = get_canonizer(cfg.model.name)

        x_latent_all = []
        for batch in tqdm(dataloader):
            x, _ = batch
            x_latent = _get_features(x, cfg.cav.layer, attribution, canonizer, cfg.cav.cav_mode, device=cfg.train.device)
            x_latent = x_latent.detach().cpu()
            x_latent_all.append(x_latent)
        x_latent_all = torch.cat(x_latent_all)

        vars = {
            "encs": x_latent_all,
            "labels": dataset.get_labels().clamp(min=0)  # type: ignore
        }
        torch.save(vars, cache_path)
        
        log.info(f"Saved extracted latents to {cache_path}.")

    return x_latent_all, labels