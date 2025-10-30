import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

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


def build_model(config: DictConfig, device: torch.device):
    model = instantiate(config.model)
    model = model.to(device)
    model.eval()
    return model


def build_dataloader(config: DictConfig) -> DataLoader:
    dataset = instantiate(config.dataset)
    return DataLoader(
        dataset,
        batch_size=config.experiment.batch_size,
        shuffle=False,
        num_workers=config.experiment.num_workers,
        drop_last=False,
    )


def run_encode(config: DictConfig):

    cache_dir = Path("variables")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "vars_celeba_diffae.pt"

    if cache_path.exists():
        log.info("Loading cached encodings from %s", cache_path)
        vars = torch.load(cache_path, map_location="cpu")
        return vars["encs"].float(), vars["labels"].float()

    experiment_cfg = config.experiment

    log.info("Seeding RNGs with %s", experiment_cfg.seed)
    seed_everything(int(experiment_cfg.seed))

    device = torch.device(experiment_cfg.device)
    log.info(f"Using device {device}")

    log.info('Building components')
    model = build_model(config, device)

    log.info('Initializing dataloader')
    dataloader = build_dataloader(config)

    log.info("Encoding entire CelebA dataset for latent extraction.")

    encodings = []

    for batch in tqdm(dataloader):

        with torch.no_grad():
            batch_x = batch[0].to(device)
            batch_encs = model.encode(batch_x).detach().cpu()

        encodings.append(batch_encs)

    encodings = torch.cat(encodings)
    labels = dataloader.dataset.get_labels().float().clamp(min=0)  # type: ignore

    vars = {
            "encs": encodings,
            "labels": labels
        }

    torch.save(vars, cache_path)
    log.info("Encodings shape: %s", encodings.shape)
    log.info("Labels shape: %s", labels.shape)
    log.info("Saved encodings to %s", cache_path)

    return encodings, labels
