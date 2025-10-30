import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchvision
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import get_dataset
from hydra.utils import instantiate

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


def build_dataset(config: DictConfig):
    dataset_cfg = OmegaConf.to_container(config.dataset, resolve=True)
    assert isinstance(dataset_cfg, dict)
    dataset_name = dataset_cfg.pop("name")
    dataset_fn = get_dataset(dataset_name)
    return dataset_fn(**dataset_cfg)


def build_dataloader(config: DictConfig) -> DataLoader:
    dataset = build_dataset(config)
    return DataLoader(
        dataset,
        batch_size=config.experiment.batch_size,
        shuffle=False,
        num_workers=config.experiment.num_workers,
        drop_last=False,
    )


def save_batch(config, batch_imgs, batch_idx):
    path_save = Path("results") / config.experiment.run_id
    for idx, img in zip(batch_idx, batch_imgs):
        path_save_spec = path_save / str(idx.item()) / f"img.{config.experiment.format}"
        path_save_spec.parent.mkdir(parents=True, exist_ok=True)
        torchvision.utils.save_image(img, str(path_save_spec))


def run_decode(config: DictConfig):
    experiment_cfg = config.experiment

    log.info("Seeding RNGs with %s", experiment_cfg.seed)
    seed_everything(int(experiment_cfg.seed))

    device = torch.device(experiment_cfg.device)
    log.info("Using device %s", device)

    log.info("Building model")
    model = build_model(config, device)

    log.info("Initializing dataloader")
    dataloader = build_dataloader(config)

    log.info("Starting generation loop")
    for batch in tqdm(dataloader):

        batch_img, batch_enc, _, batch_idx = batch
        batch_img = batch_img.to(device=device, dtype=torch.float32)
        batch_enc = batch_enc.to(device=device, dtype=torch.float32)

        with torch.no_grad():
            batch_enc_orig = model.encode(batch_img)
            batch_x_T = model.encode_stochastic(batch_img, batch_enc_orig)
            batch_dec = model.decode(batch_x_T, batch_enc)

        save_batch(config, batch_dec.cpu(), batch_idx)
