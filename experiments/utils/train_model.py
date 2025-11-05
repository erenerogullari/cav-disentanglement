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
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

from datasets import get_dataset
from hydra.utils import get_original_cwd
from models import get_fn_model_loader
from datasets import BaseDataset

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

def _resolve_checkpoint_path(cfg_model: DictConfig, dataset_name: str) -> Path:
    checkpoint_dir = Path(get_original_cwd()) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"checkpoint_{cfg_model.name}_{dataset_name}.pth"
    return checkpoint_path


def _instantiate_dataset(cfg: DictConfig) -> Tuple[BaseDataset, bool]:
    container = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(container, dict)
    dataset_name = container.pop("name")
    shuffle = container.pop("shuffle", True)
    dataset_fn = get_dataset(dataset_name)
    dataset = _call_with_matching_kwargs(dataset_fn, container)
    assert isinstance(dataset, BaseDataset)
    return dataset, shuffle


def _prepare_targets(targets, device: torch.device) -> torch.Tensor:
    if not torch.is_tensor(targets):
        targets = torch.tensor(targets)
    targets = targets.to(device)
    if targets.ndim == 0:
        targets = targets.unsqueeze(0)
    if targets.ndim == 1:
        targets = targets.unsqueeze(1)
    return targets.float()


def _multilabel_stats(outputs: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5):
    probs = torch.sigmoid(outputs).detach().cpu()
    preds = (probs >= threshold).int()
    true = targets.detach().cpu().int()

    # macro averages treat each class equally, revealing classes that are always missed
    macro_f1 = f1_score(true, preds, average="samples", zero_division=0)
    macro_recall = recall_score(true, preds, average="samples", zero_division=0)
    matches = (preds == true).float()
    per_class_acc = matches.mean(dim=0)
    macro_acc = float(per_class_acc.mean().item()) if per_class_acc.numel() else 0.0

    return {
        "f1": macro_f1,
        "recall": macro_recall,
        "accuracy": macro_acc,
    }

def _precision_recall_analysis(
    logits: torch.Tensor,   # shape (N, C), raw model outputs
    targets: torch.Tensor,  # shape (N, C), {0,1}
) -> Dict[str, object]:
    """
    Returns per-class precision/recall curves and AP, plus macro-averaged AP.
    """
    if logits.ndim != 2 or targets.ndim != 2:
        raise ValueError("Expected logits and targets to be 2D tensors of shape (N, C).")

    probs = torch.sigmoid(logits).detach().cpu().numpy()
    true = targets.detach().cpu().numpy()
    num_classes = probs.shape[1]

    curves = {}
    per_class_ap = {}
    ap_values = []

    for class_idx in range(num_classes):
        y_true = true[:, class_idx]
        y_score = probs[:, class_idx]

        if np.sum(y_true) == 0:
            precision = np.array([1.0])
            recall = np.array([0.0])
            thresholds = np.array([])
            ap = 0.0
        else:
            precision, recall, thresholds = precision_recall_curve(y_true, y_score)
            ap = float(average_precision_score(y_true, y_score))

        curves[class_idx] = {
            "precision": precision,
            "recall": recall,
            "thresholds": thresholds,
        }
        per_class_ap[class_idx] = ap
        ap_values.append(ap)

    macro_ap = float(np.mean(ap_values)) if ap_values else 0.0
    return {
        "curves": curves,
        "per_class_ap": per_class_ap,
        "macro_ap": macro_ap,
    }


def _train_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion,
    optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = _prepare_targets(targets, device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        batch_stats = _multilabel_stats(outputs.detach(), targets)
        total_acc += batch_stats["accuracy"] * batch_size
        total_samples += batch_size

    avg_loss = total_loss / total_samples
    avg_acc = total_acc / total_samples
    return avg_loss, avg_acc


def _evaluate(
    model: torch.nn.Module,
    dataloader: Optional[DataLoader],
    criterion,
    device: torch.device,
) -> Tuple[float, Dict[str, float], torch.Tensor, torch.Tensor]:
    if dataloader is None or len(dataloader) == 0:
        raise ValueError("Dataloader for evaluation is None or empty.")

    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_outputs = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = _prepare_targets(targets, device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            all_outputs.append(outputs.detach().cpu())
            all_targets.append(targets.detach().cpu())

    avg_loss = total_loss / total_samples
    logits = torch.cat(all_outputs, dim=0)
    targets_tensor = torch.cat(all_targets, dim=0)
    stats = _multilabel_stats(logits, targets_tensor)
    return avg_loss, stats, logits, targets_tensor


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


def _load_model(cfg_model: DictConfig, dataset_name: str, device: str, **override_kwargs) -> nn.Module:
    model_loader = get_fn_model_loader(cfg_model.name)
    ckpt_path = _resolve_checkpoint_path(cfg_model, dataset_name)
    loader_kwargs = {
        "ckpt_path": ckpt_path,
        "pretrained": getattr(cfg_model, "pretrained", True),
        "n_class": getattr(cfg_model, "n_class", 2) or 2,
    }
    if hasattr(cfg_model, "in_channels"):
        loader_kwargs["in_channels"] = getattr(cfg_model, "in_channels")
    if hasattr(cfg_model, "input_size"):
        loader_kwargs["input_size"] = getattr(cfg_model, "input_size")
    if override_kwargs:
        loader_kwargs.update(override_kwargs)
    if "device" in inspect.signature(model_loader).parameters:
        loader_kwargs["device"] = device
    model = _call_with_matching_kwargs(model_loader, loader_kwargs)
    assert isinstance(model, nn.Module)
    return model


@hydra.main(version_base=None, config_path="../../configs", config_name="train_model")
def run(cfg: DictConfig) -> None:
    log.info("Starting training run: %s", cfg.experiment.name)
    _set_seed(cfg.train.random_seed)

    device = cfg.train.device
    log.info("Using device: %s", device)

    log.info("Using dataset: %s", cfg.dataset.name)
    dataset, shuffle = _instantiate_dataset(cfg.dataset)
    assert isinstance(dataset, BaseDataset)
    if hasattr(dataset, "get_num_classes"):
        num_classes = dataset.get_num_classes()
    else:
        labels = dataset.get_labels() if hasattr(dataset, "get_labels") else torch.zeros((0, 0))
        if torch.is_tensor(labels) and labels.ndim >= 1:
            num_classes = labels.shape[-1] if labels.ndim > 1 else 1
        else:
            num_classes = 1

    if hasattr(dataset, "get_class_names"):
        class_names = dataset.get_class_names()
    else:
        class_names = [f"class_{i}" for i in range(num_classes)]
    log.info("Detected %d classes.", num_classes)

    if hasattr(cfg.model, "n_class"):
        cfg.model.n_class = num_classes
    else:
        cfg.model.n_class = num_classes

    train_dataset, train_loader, val_loader, test_loader = _build_loaders(dataset, shuffle, cfg.train)

    sample_tensor, _ = train_dataset[0] if len(train_dataset) > 0 else dataset[0]
    if sample_tensor.ndim == 3:
        in_channels = int(sample_tensor.shape[0])
        height, width = sample_tensor.shape[-2], sample_tensor.shape[-1]
    elif sample_tensor.ndim == 4:
        in_channels = int(sample_tensor.shape[1])
        height, width = sample_tensor.shape[-2], sample_tensor.shape[-1]
    else:
        raise ValueError(f"Unexpected input tensor shape: {sample_tensor.shape}")
    input_size = height if height == width else (height, width)

    log.info("Using model: %s", cfg.model.name)
    model = _load_model(
        cfg.model,
        cfg.dataset.name,
        device,
        in_channels=in_channels,
        input_size=input_size,
    )
    model = model.to(device)

    pos_weight = None
    if hasattr(train_dataset, "labels") and torch.is_tensor(train_dataset.labels) and train_dataset.labels.numel() > 0:
        positives = train_dataset.labels.sum(dim=0)
        total = torch.tensor(train_dataset.labels.shape[0], dtype=positives.dtype, device=positives.device)
        negatives = total - positives
        with torch.no_grad():
            pos_weight = torch.where(positives > 0, negatives / (positives + 1e-8), torch.zeros_like(positives))
    if pos_weight is not None and pos_weight.numel() > 0:
        pos_weight = pos_weight.to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay)

    best_state = None
    best_metric = 0
    best_epoch = 0

    log_every = max(1, cfg.train.log_interval)

    log.info("Starting training for %s epochs...", cfg.train.num_epochs)
    for epoch in range(1, cfg.train.num_epochs + 1):
        train_loss, train_acc = _train_epoch(model, train_loader, criterion, optimizer, device)

        if val_loader is not None:
            val_loss, val_stats, _, _ = _evaluate(model, val_loader, criterion, device)
            if cfg.train.save_best and val_stats["f1"] > best_metric:
                best_metric = val_stats["f1"]
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
            if epoch % log_every == 0 or epoch == 1 or epoch == cfg.train.num_epochs:
                log.info(
                    "Epoch %03d | train_loss=%.4f train_acc=%.4f | val_loss=%.4f val_acc=%.4f val_f1=%.4f val_recall=%.4f",
                    epoch,
                    train_loss,
                    train_acc,
                    val_loss,
                    val_stats["accuracy"],
                    val_stats["f1"],
                    val_stats["recall"],
                )
        else:
            if epoch % log_every == 0 or epoch == 1 or epoch == cfg.train.num_epochs:
                log.info("Epoch %03d | train_loss=%.4f train_acc=%.4f", epoch, train_loss, train_acc)

    if best_state is not None:
        log.info("Loading best model from epoch %d with val_f1=%.4f", best_epoch, best_metric)
        model.load_state_dict(best_state)

    test_loss, test_stats, test_logits, test_targets = _evaluate(model, test_loader, criterion, device)
    log.info(
        "Test metrics | loss=%.4f acc=%.4f f1=%.4f recall=%.4f",
        test_loss,
        test_stats["accuracy"],
        test_stats["f1"],
        test_stats["recall"],
    )

    pr_results = _precision_recall_analysis(test_logits, test_targets)
    per_class_ap_named = {
        class_names[class_idx] if class_idx < len(class_names) else f"class_{class_idx}": ap
        for class_idx, ap in pr_results["per_class_ap"].items()     # type: ignore
    }
    per_class_avg_recall = {
        class_names[class_idx] if class_idx < len(class_names) else f"class_{class_idx}": float(np.mean(curve["recall"]))     # type: ignore
        for class_idx, curve in pr_results["curves"].items()     # type: ignore
    }
    log.info("Test macro AP: %.4f", pr_results["macro_ap"])

    media_dir = Path(get_original_cwd()) / "media"
    media_dir.mkdir(parents=True, exist_ok=True)
    plt.switch_backend("Agg")
    plt.figure()
    for class_idx, curve in pr_results["curves"].items():   # type: ignore
        plt.plot(curve["recall"], curve["precision"])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves (Test)")
    plt.tight_layout()
    pr_plot_path = media_dir / f"precision_recall_{cfg.model.name}_{cfg.dataset.name}.pdf"
    plt.savefig(pr_plot_path, format="pdf")
    plt.close()
    log.info("Saved precision-recall curves to %s", pr_plot_path)

    plt.figure()
    recalls = [per_class_avg_recall[name] for name in per_class_ap_named]
    aps = [per_class_ap_named[name] for name in per_class_ap_named]
    plt.scatter(recalls, aps, s=20)
    plt.xlabel("Average Recall")
    plt.ylabel("Average Precision")
    plt.title("Per-class AP vs. Average Recall (Test)")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    ap_scatter_path = media_dir / f"precision_recall_ap_scatter_{cfg.model.name}_{cfg.dataset.name}.pdf"
    plt.savefig(ap_scatter_path, format="pdf")
    plt.close()
    log.info("Saved per-class AP scatter plot to %s", ap_scatter_path)

    checkpoint_path = _resolve_checkpoint_path(cfg.model, cfg.dataset.name)
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)
    log.info("Saved checkpoint to %s", checkpoint_path)

if __name__ == "__main__":
    run()
