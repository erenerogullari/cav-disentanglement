import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import torch.nn.functional as F
from omegaconf import DictConfig
from hydra.utils import instantiate
import numpy as np
from zennit.core import stabilize

from src import utils

import logging
log = logging.getLogger(__name__)


def get_fabric(config):
    fabric = instantiate(config.fabric)
    fabric.seed_everything(config.exp.seed)
    fabric.launch()
    return fabric


def get_components(config, fabric):
    dir_model = instantiate(config.direction_model)
    dir_model.load_state_dict(torch.load(config.exp.path_dir_model_ckpt), strict = False)
    dir_model = fabric.setup(dir_model)
    return dir_model


def get_dataloader(config, fabric):
    # return fabric.setup_dataloaders(instantiate(config.dataset))
    return instantiate(config.dataset)


def normalize(config, dir_model, x, means, stds):
    if config.exp.normalize_from_data:
        return (x - means) / stds
    else:
        return dir_model.normalize(x)


def denormalize(config, dir_model, x, means, stds):
    if config.exp.normalize_from_data:
        return x * stds + means
    else:
        return dir_model.denormalize(x)


def move_encs(config, data, w_dir, b_dir, median_logit, mean_length=None):

    if config.exp.move_method == "adaptive":
    
        step_sizes = (median_logit - b_dir - data @ w_dir) / (w_dir @ w_dir)
        return data + step_sizes.unsqueeze(1) * w_dir.unsqueeze(0).repeat(data.shape[0], 1)
    
    elif config.exp.move_method == "constant":
    
        signs = (median_logit - b_dir - data @ w_dir).sign().unsqueeze(1)
        w_dir = F.normalize(w_dir, dim = 0).unsqueeze(0).repeat(data.shape[0], 1)
        return data + signs * config.exp.step_size * data.shape[1] ** (1 / 2) * w_dir
    
    ########################  CUSTOM  ########################
    elif config.exp.move_method == "signal":

        outs = data + 0
        is_2dim = data.dim() == 2
        outs = outs[..., None, None] if is_2dim else outs
        cav = w_dir.to(outs.device)
        # length = (outs.flatten(start_dim=1) * cav).sum(1)
        length = (outs.flatten(start_dim=2).max(2).values * cav).sum(1)
        beta = (cav * cav).sum(0)
        mag = (mean_length - length).to(outs) / stabilize(beta)

        v = cav.reshape(1, *outs.shape[1:]) if config.exp.cav_mode == "full" else cav[..., None, None]
        addition = (mag[:, None, None, None] * v)
        acts = outs + addition * config.exp.step_size
        acts = acts.squeeze(-1).squeeze(-1) if is_2dim else acts
        return acts
    
    elif config.exp.move_method == "no_move":
        return data
    
    else:
        raise ValueError(f'Unknown move method: {config.exp.move_method}')
    
    ########################  ======  ########################



def run(config: DictConfig):
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f'Using device {device}')
    utils.preprocess_config(config)
    utils.setup_wandb(config)

    log.info("Launching Fabric")
    fabric = get_fabric(config)

    log.info("Initializing dataloader")
    dataloader = get_dataloader(config, fabric)

    log.info("Building components")
    dir_model = get_components(config, fabric)

    log.info("Searching for data to keep")
    label_to_id = dataloader.dataset.label_to_id
    id_to_label = dataloader.dataset.id_to_label
    label_name = config.exp.label_name
    label_id = label_to_id[label_name]

    encs_to_keep = []
    encs_to_move = []
    idx_to_move = []
    labels_to_move = []

    for batch_id, batch in tqdm(enumerate(dataloader)):
        _, batch_enc, batch_label, batch_idx = batch
        batch_label_spec = batch_label[:, label_id]
        chosen_to_move = batch_label_spec == config.exp.label_value_to_move
        encs_to_keep.append(batch_enc[~chosen_to_move])
        encs_to_move.append(batch_enc[chosen_to_move])
        idx_to_move.append(batch_idx[chosen_to_move])
        labels_to_move.append(batch_label[chosen_to_move])

    idx_to_move = torch.cat(idx_to_move).to(device)
    encs_to_keep = torch.cat(encs_to_keep).to(device)
    encs_to_move = torch.cat(encs_to_move).to(device)
    labels_to_move = torch.cat(labels_to_move).to(device)

    log.info("Computing median logit")
    means, stds = dataloader.dataset.means, dataloader.dataset.stds
    encs_to_keep = normalize(config, dir_model, encs_to_keep, means, stds)

    with torch.no_grad():
        
        w_dir, b_dir = dir_model.get_direction(label_id)  
        w_dir = w_dir.to(device)
        b_dir = b_dir.to(device)

        ########################  CUSTOM  ########################
        if config.exp.move_method == "signal":
            w_dir = F.normalize(w_dir, dim=0)
            median_logit = None
            mean_length = (encs_to_keep.flatten(start_dim=1)  * w_dir).sum(1).mean(0)
        else:
            mean_length = None
            logits = dir_model(encs_to_keep)[:, label_id]
            median_logit = logits.median()
        ########################  =====  ########################

        log.info("Editing encodings")
        encs_to_move = normalize(config, dir_model, encs_to_move, means, stds)
        encs_to_move = move_encs(config, encs_to_move, w_dir, b_dir, median_logit, mean_length)
        encs_to_move = denormalize(config, dir_model, encs_to_move, means, stds)

    log.info("Saving modified encodings")
    results = {}

    for iter, (enc, idx) in enumerate(zip(encs_to_move, idx_to_move)):
        labels = labels_to_move[iter]
        labels = {k: v.item() for k, v in zip(id_to_label, labels)}
        results[idx.item()] = {"enc": enc, "labels": labels}

    path_out = Path("results") / config.exp.run_id
    path_out.mkdir(parents = True, exist_ok = True)
    
    torch.save(results, path_out / "encodings.pt")

    

    