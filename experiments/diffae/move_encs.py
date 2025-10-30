import random
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm
from zennit.core import stabilize

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


def build_dataloader(config: DictConfig) -> DataLoader:
    dataset = instantiate(config.dataset)
    return DataLoader(
        dataset,
        batch_size=config.experiment.batch_size,
        shuffle=bool(config.dataset.shuffle),
        num_workers=config.experiment.num_workers,
        drop_last=False,
    )


def load_direction_model(config: DictConfig, device: torch.device):
    dir_model = instantiate(config.direction_model)
    state_dict = torch.load(config.experiment.direction_checkpoint, map_location="cpu")
    dir_model.load_state_dict(state_dict, strict=False)
    dir_model = dir_model.to(device)
    dir_model.eval()
    dir_model.requires_grad_(False)
    return dir_model


def normalize(experiment_cfg, dir_model, x, means, stds):
    if experiment_cfg.normalize_from_data:
        return (x - means) / stds
    else:
        return dir_model.normalize(x)


def denormalize(experiment_cfg, dir_model, x, means, stds):
    if experiment_cfg.normalize_from_data:
        return x * stds + means
    else:
        return dir_model.denormalize(x)


def move_encs(experiment_cfg, data, w_dir, b_dir, median_logit, mean_length=None):

    if experiment_cfg.move_method == "adaptive":
    
        step_sizes = (median_logit - b_dir - data @ w_dir) / (w_dir @ w_dir)
        return data + step_sizes.unsqueeze(1) * w_dir.unsqueeze(0).repeat(data.shape[0], 1)
    
    elif experiment_cfg.move_method == "constant":
    
        signs = (median_logit - b_dir - data @ w_dir).sign().unsqueeze(1)
        w_dir = F.normalize(w_dir, dim = 0).unsqueeze(0).repeat(data.shape[0], 1)
        return data + signs * experiment_cfg.step_size * data.shape[1] ** (1 / 2) * w_dir
    
    elif experiment_cfg.move_method == "signal":

        outs = data + 0
        is_2dim = data.dim() == 2
        outs = outs[..., None, None] if is_2dim else outs
        cav = w_dir.to(outs.device)
        length = (outs.flatten(start_dim=2).max(2).values * cav).sum(1)
        beta = (cav * cav).sum(0)
        mag = (mean_length - length).to(outs) / stabilize(beta)

        v = cav.reshape(1, *outs.shape[1:]) if experiment_cfg.cav_mode == "full" else cav[..., None, None]
        addition = (mag[:, None, None, None] * v)
        acts = outs + addition * experiment_cfg.step_size
        acts = acts.squeeze(-1).squeeze(-1) if is_2dim else acts
        return acts
    
    elif experiment_cfg.move_method == "no_move":
        return data
    
    else:
        raise ValueError(f'Unknown move method: {experiment_cfg.move_method}')



def run_move_encs(config: DictConfig, encodings: torch.Tensor, labels: torch.Tensor, dir_models: Dict[float, nn.Module]) -> None:

    experiment_cfg = config.experiment

    log.info("Seeding RNGs with %s", experiment_cfg.seed)
    seed_everything(int(experiment_cfg.seed))

    device = torch.device(experiment_cfg.device)
    log.info("Using device %s", device)

    log.info("Initializing dataloader")
    dataloader = build_dataloader(config)
    dataset = getattr(dataloader, "dataset", None)
    if dataset is None:
        raise AttributeError("The instantiated dataloader does not expose a dataset attribute.")

    log.info("Building components")
    dir_model = load_direction_model(config, device)

    log.info("Searching for data to keep")
    if not hasattr(dataset, "label_to_id") or not hasattr(dataset, "id_to_label"):
        raise AttributeError("Dataset must provide 'label_to_id' and 'id_to_label' attributes.")

    label_to_id = dataset.label_to_id
    id_to_label = dataset.id_to_label
    label_name = experiment_cfg.concept_name
    label_id = label_to_id[label_name]

    encs_to_keep = []
    encs_to_move = []
    idx_to_move = []
    labels_to_move = []

    for batch in tqdm(dataloader):
        _, batch_enc, batch_label, batch_idx = batch
        batch_label_spec = batch_label[:, label_id]
        chosen_to_move = batch_label_spec == experiment_cfg.concept_label_to_move
        encs_to_keep.append(batch_enc[~chosen_to_move])
        encs_to_move.append(batch_enc[chosen_to_move])
        idx_to_move.append(batch_idx[chosen_to_move])
        labels_to_move.append(batch_label[chosen_to_move])

    if not encs_to_move or not idx_to_move:
        raise RuntimeError(f"No samples found with label value {experiment_cfg.concept_label_to_move} for '{label_name}'.")

    idx_to_move = torch.cat(idx_to_move).to(device)
    encs_to_keep = torch.cat(encs_to_keep).to(device)
    encs_to_move = torch.cat(encs_to_move).to(device)
    labels_to_move = torch.cat(labels_to_move).to(device)

    log.info("Computing median logit")
    means = torch.as_tensor(dataset.means, device=device, dtype=encs_to_keep.dtype)
    stds = torch.as_tensor(dataset.stds, device=device, dtype=encs_to_keep.dtype)

    encs_to_keep = normalize(experiment_cfg, dir_model, encs_to_keep, means, stds)

    with torch.no_grad():
        
        w_dir, b_dir = dir_model.get_direction(label_id)  
        w_dir = w_dir.to(device)
        b_dir = b_dir.to(device)

        if experiment_cfg.move_method == "signal":
            w_dir = F.normalize(w_dir, dim=0)
            median_logit = None
            mean_length = (encs_to_keep.flatten(start_dim=1)  * w_dir).sum(1).mean(0)
        else:
            mean_length = None
            logits = dir_model(encs_to_keep)[:, label_id]
            median_logit = logits.median()

        log.info("Editing encodings")
        encs_to_move = normalize(experiment_cfg, dir_model, encs_to_move, means, stds)
        encs_to_move = move_encs(experiment_cfg, encs_to_move, w_dir, b_dir, median_logit, mean_length)
        encs_to_move = denormalize(experiment_cfg, dir_model, encs_to_move, means, stds)

    log.info("Saving modified encodings")
    results = {}

    for iter, (enc, idx) in enumerate(zip(encs_to_move, idx_to_move)):
        labels = labels_to_move[iter].detach().cpu()
        labels = {k: v.item() for k, v in zip(id_to_label, labels)}
        results[idx.item()] = {"enc": enc.detach().cpu(), "labels": labels}

    path_out = Path("results") / experiment_cfg.run_id
    path_out.mkdir(parents = True, exist_ok = True)
    
    torch.save(results, path_out / "encodings.pt")
