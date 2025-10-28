import math
import torch
from pathlib import Path
from tqdm import tqdm
from omegaconf import DictConfig
from hydra.utils import instantiate

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


def add_to_results(res_d, encs, labels, idxs):
    for iter, (enc, idx) in enumerate(zip(encs, idxs)):
        res_d[idx.item()] = {
            "enc": enc, 
            "labels": {k: v[iter].item() for k, v in labels.items()}}


def save_results(config, res_d, file_id):
    path_out = Path("results") / config.exp.run_id
    path_out.mkdir(parents = True, exist_ok = True)

    file_id = file_id // config.exp.save_every
    path_out_file = path_out / f"encodings_{file_id}.pt"

    if path_out_file.exists():
        path_out_file = path_out / f"encodings_{file_id + 1}.pt"

    torch.save(res_d, path_out_file)


def run_encode(config: DictConfig):

    log.info(f'Launching Fabric')
    fabric = get_fabric(config)

    log.info(f'Building components')
    model = get_components(config, fabric)

    log.info(f'Initializing dataloader')
    dataloader = get_dataloader(config, fabric)

    iter_counter = 0
    results = {}
    n_batches = math.ceil(len(dataloader.dataset) / dataloader.batch_size)

    for batch_id, batch in tqdm(enumerate(dataloader)):

        iter_counter += 1

        with torch.no_grad():
            batch_x, batch_labels, batch_idx = batch
            batch_encs = model.encode(batch_x)
            
        assert batch_idx.unique().shape[0] == batch_x.shape[0]
        add_to_results(results, batch_encs, batch_labels, batch_idx)

        if iter_counter % config.exp.save_every == 0 or iter_counter == n_batches:
            save_results(config, results, batch_id)
            results = {}
