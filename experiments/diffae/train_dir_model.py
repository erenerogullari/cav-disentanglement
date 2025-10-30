import random
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from experiments.utils.utils import initialize_weights

import logging
log = logging.getLogger(__name__)


def seed_everything(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def find_cached_dir_models(config: DictConfig, alphas: list, cache_dir: Path) -> Dict[float, nn.Module]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    dir_models = {}
    for alpha in alphas:    
        cache_path = cache_dir / f"{config.dir_model.name}:alpha{alpha}.pt"
        if cache_path.exists():
            log.info(f"Found cached CAVs for alpha={alpha} in {cache_dir}.")
            dir_model = torch.load(cache_path, map_location="cpu")
            if hasattr(dir_model, "eval"):
                dir_model.eval()
            dir_models[alpha] = dir_model

    return dir_models


def build_dataloader(encodings, labels, batch_size: int) -> DataLoader:
    dataset = torch.utils.data.TensorDataset(encodings, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def train_dir_model(config: DictConfig, encodings: torch.Tensor, labels: torch.Tensor) -> Dict[float, nn.Module]:

    alphas = config.dir_model.alphas
    cache_dir = Path("results") / "diffae" / "dir_models"
    dir_models = find_cached_dir_models(config, alphas, cache_dir)
    alphas_to_train = [alpha for alpha in alphas if alpha not in dir_models.keys()]

    if len(alphas_to_train) == 0:
        log.info("All CAVs found in cache. Skipping training.")
        return dir_models

    log.info("Seeding RNGs with %s", config.experiment.seed)
    seed_everything(int(config.experiment.seed))

    device = torch.device(config.experiment.device)
    log.info(f"Using device {device}")


    log.info('Initializing dataloader')
    encodings = encodings.float()
    labels = labels.float()
    dataloader = build_dataloader(encodings, labels, config.experiment.batch_size)

    n_concepts = labels.shape[1]
    n_features = encodings.shape[1]

    for alpha in alphas_to_train:
        log.info(f"Training direction model for alpha={alpha}")

        log.info('Building components')
        cav_cfg = {"_target_": config.dir_model["_target_"]}
        dir_model = instantiate(
            cav_cfg,
            n_concepts=n_concepts,
            n_features=n_features,
            device=device,
        )
        dir_model = dir_model.to(device)
        optimizer = optim.Adam(dir_model.parameters(), lr=config.dir_model.learning_rate)

        for epoch in tqdm(range(config.dir_model.n_epochs), desc="Epochs"):
            dir_model.train()
            total_cav_loss = 0.0
            total_orth_loss = 0.0
            num_batches = 0

            for batch_encs, batch_labels in dataloader:
                batch_encs = batch_encs.to(device=device, dtype=torch.float32)
                batch_labels = batch_labels.to(device=device, dtype=torch.float32).clamp(min=0)

                optimizer.zero_grad()
                cav_loss, orth_loss = dir_model.train_step(batch_encs, batch_labels, alpha)
                loss = cav_loss + orth_loss
                loss.backward()
                optimizer.step()

                total_cav_loss += cav_loss.item()
                total_orth_loss += orth_loss.item()
                num_batches += 1

            if (epoch + 1) % 25 == 0 or epoch == config.dir_model.n_epochs - 1:
                log.info(
                    "epoch=%d cav_loss=%.5f orth_loss=%.5f",
                    epoch,
                    total_cav_loss / max(num_batches, 1),
                    total_orth_loss / max(num_batches, 1),
                )

        dir_model_cpu = dir_model.to("cpu")
        dir_model_cpu.eval()
        cache_path = cache_dir / f"{config.dir_model.name}:alpha{alpha}.pt"
        torch.save(dir_model_cpu, cache_path)
        dir_models[alpha] = dir_model_cpu

    return dir_models
