import random
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import Dataset
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

def _format_float(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    text = f"{value}"
    if "e" in text or "E" in text:
        return f"{value:.4g}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def _cache_hit(cache_dir: Path, step_sizes: list) -> bool:
        for step in step_sizes:
            suffix = _format_float(step) if step is not None else "0"
            target = cache_dir / f"step_size{suffix}" / "encodings.pt"
            if not target.exists():
                return False
        return True


def get_target_encodings(move_cfg: DictConfig, dataset: Dataset, labels: torch.Tensor, target_label: int, target_concept: str):
    label_to_id = dataset.label_to_id # type: ignore
    if target_concept not in label_to_id.keys():
        raise ValueError(f"Concept '{target_concept}' not found in dataset concepts.")
    
    select_label = 1 - target_label
    concept_idx = label_to_id[target_concept]
    idx_to_move = (labels[:, concept_idx] == select_label).nonzero(as_tuple=True)[0]

    if idx_to_move.size(0) == 0:
        raise RuntimeError(f"No samples found for concept '{target_concept}' with label '{select_label}'.")

    return idx_to_move, concept_idx


def compute_stats(encodings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    means = encodings.mean(dim=0)
    stds = encodings.std(dim=0)
    return means, stds


def normalize(move_cfg, dir_model, x, means, stds):
    if move_cfg.normalize_from_data:
        return (x - means) / stds
    else:
        return dir_model.normalize(x)


def denormalize(move_cfg, dir_model, x, means, stds):
    if move_cfg.normalize_from_data:
        return x * stds + means
    else:
        return dir_model.denormalize(x)


def move_encs(move_cfg, data, w_dir, b_dir, median_logit, mean_length=None, step_size: Optional[float] = None):

    if move_cfg.move_method == "adaptive":
    
        step_sizes = (median_logit - b_dir - data @ w_dir) / (w_dir @ w_dir)
        return data + step_sizes.unsqueeze(1) * w_dir.unsqueeze(0).repeat(data.shape[0], 1)
    
    elif move_cfg.move_method == "constant":

        if step_size is None:
            raise ValueError("step_size must be provided when using 'constant' move_method.")
    
        signs = (median_logit - b_dir - data @ w_dir).sign().unsqueeze(1)
        w_dir = F.normalize(w_dir, dim = 0).unsqueeze(0).repeat(data.shape[0], 1)
        return data + signs * step_size * data.shape[1] ** (1 / 2) * w_dir
    
    elif move_cfg.move_method == "clarc":

        if step_size is None:
            raise ValueError("step_size must be provided when using 'clarc' move_method.")

        outs = data + 0
        is_2dim = data.dim() == 2
        outs = outs[..., None, None] if is_2dim else outs
        cav = w_dir.to(outs.device)
        length = (outs.flatten(start_dim=2).max(2).values * cav).sum(1)
        beta = (cav * cav).sum(0)
        mag = (mean_length - length).to(outs) / stabilize(beta)

        v = cav.reshape(1, *outs.shape[1:]) if move_cfg.cav_mode == "full" else cav[..., None, None]
        addition = (mag[:, None, None, None] * v)
        acts = outs + addition * step_size
        acts = acts.squeeze(-1).squeeze(-1) if is_2dim else acts
        return acts
    
    else:
        raise ValueError(f"Unknown move method: {move_cfg.move_method}")



def run_move_encs(config: DictConfig, encodings: torch.Tensor, labels: torch.Tensor, dir_model: nn.Module) -> None:

    experiment_cfg = config.experiment
    move_cfg = config.move_encs
    alpha = config.dir_model.alpha 
    cache_dir = Path("results") / "diffae" / str(config.encode.dataset.name) / "moved_encs" / str(config.dir_model.name) / f"alpha{alpha}"

    cache_dir.mkdir(parents=True, exist_ok=True)

    step_sizes = getattr(move_cfg, "step_sizes", None)
    if step_sizes is None:
        raise ValueError("step_sizes must be provided in move_encs config.")
    else:
        step_sizes = [float(step) for step in step_sizes]

    if _cache_hit(cache_dir, step_sizes):
        log.info("Encodings already moved, skipping movement step.")
        return

    log.info("Seeding RNGs with %s", experiment_cfg.seed)
    seed_everything(int(experiment_cfg.seed))

    device = torch.device(experiment_cfg.device)
    log.info("Using device %s", device)

    log.info("Finding data to move")
    dataset = instantiate(move_cfg.dataset)
    id_to_label = dataset.id_to_label
    idx_to_move, concept_idx = get_target_encodings(
        move_cfg=move_cfg,
        dataset=dataset,
        labels=labels.clamp(0),
        target_label=move_cfg.target_label,
        target_concept=move_cfg.target_concept,
    )
    
    encs_to_move = encodings[idx_to_move].to(device)
    labels_to_move = labels[idx_to_move].to(device)
    encs_to_keep = encodings[~torch.isin(torch.arange(encodings.shape[0]), idx_to_move)].to(device)
    log.info("Found %s samples with concept label %s for concept %s to move.", 
             len(idx_to_move), 
             1 - move_cfg.target_label,
             move_cfg.target_concept,
             )
    

    means, stds = compute_stats(encodings.to(device))
    # moved_encodings: Dict[Optional[float], torch.Tensor] = {}

    with torch.no_grad():
        log.info("Moving encodings with direction model alpha=%s", alpha)
        dir_model = dir_model.to(device)
        dir_model.eval()
        dir_model.requires_grad_(False)

        encs_to_keep_nrm = normalize(move_cfg, dir_model, encs_to_keep, means, stds)

        w_dir, b_dir = dir_model.get_direction(concept_idx)    # type: ignore
        w_dir = w_dir.to(device)
        b_dir = b_dir.to(device)

        if move_cfg.move_method == "clarc":
            w_dir = F.normalize(w_dir, dim=0)
            mean_length = (encs_to_keep_nrm.flatten(start_dim=1) * w_dir).sum(1).mean(0)
            median_logit = None
        else:
            mean_length = None
            logits = dir_model(encs_to_keep_nrm)[:, concept_idx]
            median_logit = logits.median()

        for step_size in step_sizes:
            if move_cfg.move_method in ["constant", "clarc"] and step_size is None:
                raise ValueError(f"'step_sizes' must be provided for move_method '{move_cfg.move_method}'.")
            
            log.info("Moving encodings for step size=%s", step_size)
            encs_to_move_nrm = normalize(move_cfg, dir_model, encs_to_move, means, stds)
            moved_norm = move_encs(
                move_cfg,
                encs_to_move_nrm,
                w_dir,
                b_dir,
                median_logit,
                mean_length,
                step_size=step_size,
            )
            moved = denormalize(move_cfg, dir_model, moved_norm, means, stds)
            # moved_encodings[step_size] = moved.detach().cpu()

            log.info("Saving modified encodings")
            results = {}

            for iter, (enc, idx) in enumerate(zip(moved, idx_to_move)):
                labels = labels_to_move[iter]
                labels = {k: v.item() for k, v in zip(id_to_label, labels)} # type: ignore
                results[idx.item()] = {"enc": enc, "labels": labels}

            step_suffix = _format_float(step_size) if step_size is not None else "0"
            path_out =  cache_dir / f"step_size{step_suffix}"
            path_out.mkdir(parents = True, exist_ok = True)
            
            torch.save(results, path_out / "encodings.pt")
