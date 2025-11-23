import wandb
import torch
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from omegaconf import DictConfig
from hydra.utils import instantiate

from src import utils

import logging
log = logging.getLogger(__name__)


def get_fabric(config):
    fabric = instantiate(config.fabric)
    fabric.seed_everything(config.exp.seed)
    fabric.launch()
    return fabric


def get_components(config, fabric):
    model = fabric.setup(instantiate(config.model))
    return model


def get_dataloader(config, fabric):
    return fabric.setup_dataloaders(instantiate(config.dataset))


def save_batch(config, batch_imgs, batch_idx):
    path_save = Path("results") / config.exp.run_id
    for idx, img in zip(batch_idx, batch_imgs):
        path_save_spec = path_save / str(idx.item()) / f"img.{config.exp.format}"
        path_save_spec.parent.mkdir(parents=True, exist_ok=True)
        torchvision.utils.save_image(img, str(path_save_spec))


def run(config: DictConfig):
    utils.preprocess_config(config)
    utils.setup_wandb(config)

    log.info(f'Launching Fabric')
    fabric = get_fabric(config)

    log.info(f'Building components')
    model = get_components(config, fabric)

    log.info(f'Initializing dataloader')
    dataloader = get_dataloader(config, fabric)

    # log.info(f'Taking the subset')
    # dataset = dataloader.dataset
    # idx = 38    # Index corresponding to 'wearing necktie' concept
    # indices = (dataset.ys[:,idx] == 1).nonzero(as_tuple=True)[0].tolist()
    # subset_data = dataset.get_subset_by_idxs(indices)
    # dataloader = DataLoader(subset_data, batch_size=32)

    log.info(f'Starting generation loop')
    for batch_id, batch in tqdm(enumerate(dataloader)):

        batch_img, batch_enc, _, batch_idx = batch
        batch_img = batch_img.to(torch.float32)
        batch_enc= batch_enc.to(torch.float32)
            
        with torch.no_grad():

            batch_enc_orig = model.encode(batch_img)
            batch_x_T = model.encode_stochastic(batch_img, batch_enc_orig)
            batch_dec = model.decode(batch_x_T, batch_enc)

            if batch_id == 0:
                wandb.log({"images/original": wandb.Image(batch_img),
                       "images/modified": wandb.Image(batch_dec)})

            save_batch(config, batch_dec, batch_idx)