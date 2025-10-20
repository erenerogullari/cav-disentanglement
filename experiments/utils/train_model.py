import copy
import inspect
import logging
import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from datasets import get_dataset
from hydra.utils import get_original_cwd
from models import get_fn_model_loader

log = logging.getLogger(__name__)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _call_with_matching_kwargs(fn, kwargs: Dict) -> object:
    signature = inspect.signature(fn)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
        return fn(**kwargs)
    allowed = {
        name
        for name, param in signature.parameters.items()
        if param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    filtered = {k: v for k, v in kwargs.items() if k in allowed}
    return fn(**filtered)


def _instantiate_dataset(cfg: DictConfig) -> Tuple[object, bool]:
    container = OmegaConf.to_container(cfg, resolve=True)
    dataset_name = container.pop("name")
    shuffle = container.pop("shuffle", True)
    dataset_fn = get_dataset(dataset_name)
    dataset = _call_with_matching_kwargs(dataset_fn, container)
    return dataset, shuffle


def _prepare_targets(targets, target_index: int, device: torch.device) -> torch.Tensor:
    if not torch.is_tensor(targets):
        targets = torch.tensor(targets)
    targets = targets.to(device)
    if targets.ndim == 0:
        targets = targets.unsqueeze(0)
    if targets.ndim == 1:
        targets = targets.unsqueeze(1)
    binary = targets[:, target_index].clamp(min=0, max=1)
    return binary.round().long()


def _classification_accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    preds = outputs.argmax(dim=1)
    if targets.ndim > 1:
        targets = targets.squeeze(dim=-1)
    targets = targets.long()
    return (preds == targets).float().sum().item()


def _train_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion,
    optimizer,
    target_index: int,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0.0
    total_samples = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = _prepare_targets(targets, target_index, device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        total_correct += _classification_accuracy(outputs.detach(), targets)
        total_samples += batch_size

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


def _evaluate(
    model: torch.nn.Module,
    dataloader: Optional[DataLoader],
    criterion,
    target_index: int,
    device: torch.device,
) -> Tuple[float, float]:
    if dataloader is None or len(dataloader) == 0:
        return float("nan"), float("nan")

    model.eval()
    total_loss = 0.0
    total_correct = 0.0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = _prepare_targets(targets, target_index, device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            total_correct += _classification_accuracy(outputs, targets)
            total_samples += batch_size

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


def _build_loaders(dataset, shuffle: bool, cfg_train: DictConfig):
    train_ids, val_ids, test_ids = dataset.do_train_val_test_split(cfg_train.val_split, cfg_train.test_split)

    train_dataset = dataset.get_subset_by_idxs(train_ids)
    train_dataset.do_augmentation = True

    collate = dict(
        batch_size=cfg_train.batch_size,
        num_workers=cfg_train.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    train_loader = DataLoader(train_dataset, shuffle=shuffle, **collate)
    val_loader = DataLoader(dataset.get_subset_by_idxs(val_ids), shuffle=False, **collate) if len(val_ids) > 0 else None
    test_loader = DataLoader(dataset.get_subset_by_idxs(test_ids), shuffle=False, **collate) if len(test_ids) > 0 else None
    return train_dataset, train_loader, val_loader, test_loader


def _load_model(cfg_model: DictConfig, dataset_name: str, device: str) -> torch.nn.Module:
    model_loader = get_fn_model_loader(cfg_model.name)
    ckpt_paths = OmegaConf.to_container(cfg_model.ckpt_paths, resolve=True) if hasattr(cfg_model, "ckpt_paths") else {}
    ckpt_path = ckpt_paths.get(dataset_name) if isinstance(ckpt_paths, Dict) else None
    loader_kwargs = {
        "ckpt_path": ckpt_path,
        "pretrained": getattr(cfg_model, "pretrained", True),
        "n_class": getattr(cfg_model, "n_class", 2) or 2,
    }
    if "device" in inspect.signature(model_loader).parameters:
        loader_kwargs["device"] = device
    model = _call_with_matching_kwargs(model_loader, loader_kwargs)
    return model


@hydra.main(version_base=None, config_path="../../configs", config_name="train_model")
def run(cfg: DictConfig) -> None:
    log.info("Starting training run: %s", cfg.experiment.name)
    _set_seed(cfg.train.random_seed)

    device = cfg.train.device
    log.info("Using device: %s", device)

    log.info("Using dataset: %s", cfg.dataset.name)
    dataset, shuffle = _instantiate_dataset(cfg.dataset)
    concept_names = dataset.get_concept_names() if hasattr(dataset, "get_concept_names") else []

    task_cfg = cfg.get("task") or {}
    if "target_concept_idx" in task_cfg:
        target_index = int(task_cfg.target_concept_idx)
    elif "target_concept_name" in task_cfg:
        target_name = task_cfg.target_concept_name
        if target_name not in concept_names:
            raise ValueError(f"Target concept '{target_name}' not found. Available concepts: {concept_names}")
        target_index = concept_names.index(target_name)
    else:
        raise ValueError("Provide task.target_concept_idx or task.target_concept_name for binary classification")

    if target_index < 0 or (concept_names and target_index >= len(concept_names)):
        raise ValueError(f"target concept index {target_index} out of range for {len(concept_names)} concepts")

    target_name = concept_names[target_index] if concept_names else f"concept_{target_index}"
    log.info("Training binary classifier for concept: %s", target_name)

    train_dataset, train_loader, val_loader, test_loader = _build_loaders(dataset, shuffle, cfg.train)

    log.info("Using model: %s", cfg.model.name)
    model = _load_model(cfg.model, cfg.dataset.name, device)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay)

    best_state = None
    best_metric = float("inf") if val_loader is not None else None

    log_every = max(1, cfg.train.log_interval)

    log.info("Starting training for %s epochs...", cfg.train.num_epochs)
    for epoch in range(1, cfg.train.num_epochs + 1):
        train_loss, train_acc = _train_epoch(model, train_loader, criterion, optimizer, target_index, device)

        if val_loader is not None:
            val_loss, val_acc = _evaluate(model, val_loader, criterion, target_index, device)
            if cfg.train.save_best and val_loss < best_metric:
                best_metric = val_loss
                best_state = copy.deepcopy(model.state_dict())
            if epoch % log_every == 0 or epoch == 1 or epoch == cfg.train.num_epochs:
                log.info(
                    "Epoch %03d | train_loss=%.4f train_acc=%.4f | val_loss=%.4f val_acc=%.4f",
                    epoch,
                    train_loss,
                    train_acc,
                    val_loss,
                    val_acc,
                )
        else:
            if epoch % log_every == 0 or epoch == 1 or epoch == cfg.train.num_epochs:
                log.info("Epoch %03d | train_loss=%.4f train_acc=%.4f", epoch, train_loss, train_acc)

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc = _evaluate(model, test_loader, criterion, target_index, device)
    log.info("Test metrics | loss=%.4f acc=%.4f | concept=%s", test_loss, test_acc, target_name)

    checkpoint_dir = Path(get_original_cwd()) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"checkpoint_{cfg.model.name}_{cfg.dataset.name}.pth"
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)
    log.info("Saved checkpoint to %s", checkpoint_path)


if __name__ == "__main__":
    run()
