import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import math
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
    dataset = instantiate(config.encode.dataset)
    return DataLoader(
        dataset,
        batch_size=config.experiment.batch_size,
        shuffle=False,
        num_workers=config.experiment.num_workers
    )

def add_to_results(res_d, encs, labels, idxs):
    for iter, (enc, idx) in enumerate(zip(encs, idxs)):
        res_d[idx.item()] = {
            "enc": enc, 
            "labels": {k: v[iter].item() for k, v in labels.items()}}


def save_results(config, results, file_id):
    path_out = Path(config.decode.dataset.path_encodings)
    path_out.mkdir(parents = True, exist_ok = True)

    file_id = file_id // config.encode.save_every
    path_out_file = path_out / f"encodings_{file_id}.pt"

    if path_out_file.exists():
        path_out_file = path_out / f"encodings_{file_id + 1}.pt"

    torch.save(results, path_out_file)


def load_cached_files(cache_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    cache_files = sorted(cache_path.glob("encodings_*.pt"))
    if not cache_files:
        raise FileNotFoundError(f"No cached encodings found under {cache_path}.")

    entries: list[tuple[int, torch.Tensor, torch.Tensor]] = []
    label_names: Optional[list[str]] = None

    for cache_file in cache_files:
        chunk = torch.load(cache_file, map_location="cpu")
        for idx, payload in chunk.items():
            enc = payload["enc"].detach().cpu()
            sample_labels = payload["labels"]
            if label_names is None:
                label_names = list(sample_labels.keys())
            ordered_labels = torch.tensor(
                [sample_labels[name] for name in label_names],
                dtype=torch.float32,
            )
            entries.append((idx, enc, ordered_labels))

    entries.sort(key=lambda item: item[0])

    encodings = torch.stack([entry[1] for entry in entries], dim=0)
    labels = torch.stack([entry[2] for entry in entries], dim=0)

    return encodings, labels


def run_encode(config: DictConfig):

    cache_dir = Path(config.move_encs.dataset.path_encodings)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_files = list(cache_dir.glob("encodings_*.pt"))
    if cache_files:
        log.info("Loading cached encodings from %s", cache_dir)
        return load_cached_files(cache_dir)

    experiment_cfg = config.experiment

    log.info("Seeding RNGs with %s", experiment_cfg.seed)
    seed_everything(int(experiment_cfg.seed))

    device = torch.device(experiment_cfg.device)
    log.info(f"Using device {device}")

    log.info('Building components')
    model = build_model(config, device)

    log.info('Initializing dataloader')
    dataloader = build_dataloader(config)

    log.info("Encoding samples")

    iter_counter = 0
    results = {}
    encodings = []
    labels = []
    label_names: Optional[list[str]] = None
    n_batches = math.ceil(len(dataloader.dataset) / dataloader.batch_size)  # type: ignore

    for batch_id, batch in tqdm(enumerate(dataloader)):

        iter_counter += 1

        with torch.no_grad():
            batch_x, batch_labels, batch_idx = batch
            batch_encs = model.encode(batch_x.to(device))
            
        assert batch_idx.unique().shape[0] == batch_x.shape[0]
        add_to_results(results, batch_encs, batch_labels, batch_idx)
        encodings.append(batch_encs.cpu())

        if label_names is None:
            label_names = list(batch_labels.keys())

        ordered_batch_labels = []
        for name in label_names:
            label_column = batch_labels[name]
            if not torch.is_tensor(label_column):
                label_column = torch.tensor(label_column)
            label_column = label_column.detach().cpu().flatten().float()
            ordered_batch_labels.append(label_column)

        labels.append(torch.stack(ordered_batch_labels, dim=1))

        if iter_counter % config.encode.save_every == 0 or iter_counter == n_batches:
            save_results(config, results, batch_id)
            results = {}

    log.info("Encoding completed.")

    return torch.cat(encodings, dim=0), torch.cat(labels, dim=0)
