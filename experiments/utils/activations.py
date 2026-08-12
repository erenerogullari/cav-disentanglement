import os
import torch
import torch.nn as nn
import numpy as np
import pickle
from omegaconf import DictConfig, OmegaConf
from typing import Dict, List, Tuple, Optional
from crp.attribution import CondAttribution
from zennit import canonizers
from zennit.composites import EpsilonPlusFlat
from datasets import get_dataset
from models import (
    get_fn_model_loader,
    get_canonizer,
    requires_lxt_localization,
)
from hydra.utils import instantiate
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from hydra.utils import get_original_cwd
from experiments.utils.utils import get_dataset_cache_namespace
import logging

log = logging.getLogger(__name__)


def limit_preprocessing_dataset(cfg: DictConfig, dataset):
    """Optionally cap a real dataset for fast, opt-in smoke tests."""
    max_samples = cfg.train.get("max_preprocessing_samples", None)
    if max_samples is None:
        return dataset
    max_samples = int(max_samples)
    if max_samples <= 0:
        raise ValueError("train.max_preprocessing_samples must be positive.")
    if len(dataset) <= max_samples:
        return dataset

    labels = dataset.get_labels().clamp(min=0)
    rng = np.random.default_rng(int(cfg.train.random_seed))
    selected: set[int] = set()
    for concept_id in range(labels.shape[1]):
        for value in (0, 1):
            candidates = torch.where(labels[:, concept_id] == value)[0].numpy()
            if len(candidates):
                selected.add(int(rng.choice(candidates)))
    if len(selected) > max_samples:
        raise ValueError(
            "train.max_preprocessing_samples is too small to cover positive and "
            f"negative examples for every concept (need at least {len(selected)})."
        )
    remaining = np.setdiff1d(np.arange(len(dataset)), np.array(sorted(selected)))
    num_remaining = max_samples - len(selected)
    if num_remaining:
        selected.update(
            int(value)
            for value in rng.choice(remaining, size=num_remaining, replace=False)
        )

    subset = dataset.get_subset_by_idxs(sorted(selected))
    from datasets.base_dataset import BaseDataset

    train_ids, val_ids, test_ids = BaseDataset.do_train_val_test_split(
        subset,
        val_split=cfg.train.val_ratio,
        test_split=cfg.train.test_ratio,
        seed=cfg.train.random_seed,
    )
    subset.idxs_train = train_ids
    subset.idxs_val = val_ids
    subset.idxs_test = test_ids
    log.info(
        "Limited preprocessing dataset to %d real samples for this smoke run.",
        len(subset),
    )
    return subset


def _get_features(batch, layer_name, attribution, composite, cav_mode, device):
    if cav_mode not in {"full", "max", "avg"}:
        raise ValueError(
            f"Invalid cav_mode: {cav_mode}. Choose from 'full', 'max', or 'avg'."
        )
    batch.requires_grad = True
    dummy_cond = [{"y": 0} for _ in range(len(batch))]
    attr = attribution(
        batch.to(device), dummy_cond, composite, record_layer=[layer_name]
    )
    acts = attr.activations[layer_name]
    if acts.ndim <= 2:
        return acts
    if cav_mode == "full":
        features = acts
    elif cav_mode == "max":
        features = acts.flatten(start_dim=2).max(2)[0]
    elif cav_mode == "avg":
        features = acts.flatten(start_dim=2).mean(2)
    return features


def get_features(batch, config, attribution):

    batch.requires_grad = True
    dummy_cond = [{"y": 0} for _ in range(len(batch))]
    record_layer = [config["layer_name"]]
    attr = attribution(
        batch.to(config["device"]), dummy_cond, record_layer=record_layer
    )
    if config["cav_mode"] == "cavs_full":
        features = attr.activations[config["layer_name"]]
    else:
        # ViT support
        acts = attr.activations[config["layer_name"]]
        acts = acts if acts.dim() > 2 else acts[..., None, None]
        acts = (
            acts.transpose(1, 3).transpose(2, 3)
            if "swin_former" in config["model_name"]
            else acts
        )
        features = acts.flatten(start_dim=2).max(2)[0]
        # features = attr.activations[config["layer_name"]].flatten(start_dim=2).max(2)[0]
    return features


def extract_latents(
    cfg: DictConfig, model: nn.Module, dataset: torch.utils.data.Dataset
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract latent representations from a specified layer of the model for the entire dataset.
    Args:
        cfg (DictConfig): Configuration object containing model and dataset parameters.
        model (nn.Module): The neural network model from which to extract features.
        dataset (torch.utils.data.Dataset): The dataset for which to extract features.
    Returns:
        torch.Tensor: A tensor containing the extracted latent representations.
        torch.Tensor: A tensor containing the corresponding labels.
    """
    cache_dir = (
        Path(get_original_cwd())
        / "variables"
        / get_dataset_cache_namespace(cfg.dataset)
        / f"{cfg.model.name}"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_name = f"{cfg.cav.layer}.pth"
    cache_path = cache_dir / cache_name
    labels = dataset.get_labels().clamp(min=0)  # type: ignore

    if cache_path.exists():
        log.info(f"Loading cached latents from {cache_path}.")
        vars = torch.load(cache_path, weights_only=True)
        x_latent_all = vars["encs"]
        cached_labels = vars.get("labels")
        cache_matches_dataset = (
            x_latent_all.shape[0] == labels.shape[0]
            and isinstance(cached_labels, torch.Tensor)
            and torch.equal(cached_labels, labels)
        )
        if cache_matches_dataset:
            return x_latent_all, labels

        cached_label_shape = (
            tuple(cached_labels.shape)
            if isinstance(cached_labels, torch.Tensor)
            else None
        )
        log.warning(
            "Ignoring incompatible latent cache at %s: cached encodings/labels "
            "have shapes %s/%s, current labels have shape %s.",
            cache_path,
            tuple(x_latent_all.shape),
            cached_label_shape,
            tuple(labels.shape),
        )

    log.info("Extracting latents...")
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=False,
    )

    if requires_lxt_localization(cfg.model.name):
        import zennit.rules as z_rules
        from zennit.composites import LayerMapComposite

        composite = LayerMapComposite(
            [
                (torch.nn.Conv2d, z_rules.Gamma(100)),
                (torch.nn.Linear, z_rules.Gamma(0.1)),
            ],
            # canonizers=canonizers,
        )

    else:
        canonizers = get_canonizer(cfg.model.name)
        composite = EpsilonPlusFlat(canonizers=canonizers)

    attribution = CondAttribution(model)

    x_latent_all = []
    for batch in tqdm(dataloader):
        x, _ = batch
        x_latent = _get_features(
            x,
            cfg.cav.layer,
            attribution,
            composite,
            cfg.cav.cav_mode,
            device=cfg.train.device,
        )
        x_latent = x_latent.detach().cpu()
        x_latent_all.append(x_latent)
    x_latent_all = torch.cat(x_latent_all)

    vars = {"encs": x_latent_all, "labels": labels}
    torch.save(vars, cache_path)

    log.info(f"Saved extracted latents to {cache_path}.")

    return x_latent_all, labels
