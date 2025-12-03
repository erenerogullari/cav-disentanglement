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
from sklearn.metrics import (
    f1_score,
    recall_score,
    accuracy_score,
    precision_score,
    precision_recall_curve,
    average_precision_score,
)
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


def _multilabel_stats(outputs: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, per_class: bool = False):
    probs = torch.sigmoid(outputs).detach().cpu()
    preds = (probs >= threshold).int()
    true = targets.detach().cpu().int()
    matches = (preds == true).float()
    per_class_acc = matches.mean(dim=0)

    # macro averages treat each class equally, revealing classes that are always missed
    average = None if per_class else "macro"
    f1 = f1_score(true, preds, average=average, zero_division=0)
    precision = precision_score(true, preds, average=average, zero_division=0)
    recall = recall_score(true, preds, average=average, zero_division=0)
    acc = per_class_acc if per_class else float(per_class_acc.mean().item())

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _precision_recall_analysis(
    logits: torch.Tensor,   # shape (N, C), raw model outputs
    targets: torch.Tensor,  # shape (N, C), {0,1}
) -> Dict[str, object]:
    """
    Returns per-class precision/recall curves.
    """
    if logits.ndim != 2 or targets.ndim != 2:
        raise ValueError("Expected logits and targets to be 2D tensors of shape (N, C).")

    probs = torch.sigmoid(logits).detach().cpu().numpy()
    true = targets.detach().cpu().numpy()
    num_classes = probs.shape[1]

    curves = {}
    for class_idx in range(num_classes):
        y_true = true[:, class_idx]
        y_score = probs[:, class_idx]

        if np.sum(y_true) == 0:
            precision = np.array([1.0])
            recall = np.array([0.0])
            thresholds = np.array([])
        else:
            precision, recall, thresholds = precision_recall_curve(y_true, y_score)

        curves[class_idx] = {
            "precision": precision,
            "recall": recall,
            "thresholds": thresholds,
        }

    return curves


def _plot_precision_recall_curves(pr_curves, media_dir: Path, model_name: str, dataset_name: str) -> Path:
    plt.figure()
    for class_idx, curve in pr_curves.items():   # type: ignore
        plt.plot(curve["recall"], curve["precision"])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves (Test)")
    plt.tight_layout()
    pr_plot_path = media_dir / f"precision_recall_curves_{model_name}_{dataset_name}.pdf"
    plt.savefig(pr_plot_path, format="pdf")
    plt.close()
    return pr_plot_path


def _plot_per_class_precision_recall_scatter(
    precisions: np.ndarray,
    recalls: np.ndarray,
    media_dir: Path,
    model_name: str,
    dataset_name: str,
) -> Path:
    plt.figure()
    plt.scatter(recalls, precisions, s=20)
    plt.xlabel("Recall (threshold=0.5)")
    plt.ylabel("Precision (threshold=0.5)")
    plt.title("Per-class Precision vs. Recall (Test)")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    pr_scatter_path = media_dir / f"precision_recall_scatter_{model_name}_{dataset_name}.pdf"
    plt.savefig(pr_scatter_path, format="pdf")
    plt.close()
    return pr_scatter_path


def _train_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion,
    optimizer,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total_f1 = 0.0
    total_prec = 0.0
    total_recall = 0.0
    total_samples = 0

    train_stats = {}
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
        batch_stats = _multilabel_stats(outputs.detach(), targets, per_class=False)
        total_acc += batch_stats["accuracy"] * batch_size
        total_f1 += batch_stats["f1"] * batch_size
        total_prec += batch_stats["precision"] * batch_size
        total_recall += batch_stats["recall"] * batch_size
        total_samples += batch_size

    avg_loss = total_loss / total_samples
    train_stats["accuracy"] = total_acc / total_samples
    train_stats["f1"] = total_f1 / total_samples
    train_stats["precision"] = total_prec / total_samples
    train_stats["recall"] = total_recall / total_samples
    return avg_loss, train_stats


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
    stats = _multilabel_stats(logits, targets_tensor, per_class=False)
    return avg_loss, stats, logits, targets_tensor


def _build_loaders(dataset: BaseDataset, shuffle: bool, cfg_train: DictConfig, seed: int):
    train_ids, val_ids, test_ids = dataset.do_train_val_test_split(cfg_train.val_split, cfg_train.test_split, seed=seed)

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
        "ckpt_path": getattr(cfg_model, "ckpt_path", None),
        "pretrained": cfg_model.pretrained,
        "n_class": cfg_model.n_class,
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
    seed = cfg.train.random_seed
    _set_seed(seed)

    device = cfg.train.device
    log.info("Using device: %s", device)

    log.info("Using dataset: %s", cfg.dataset.name)
    dataset, shuffle = _instantiate_dataset(cfg.dataset)
    assert isinstance(dataset, BaseDataset)
    num_classes = dataset.get_num_classes()
    class_names = dataset.get_class_names()
    log.info("Detected %d classes.", num_classes)
    cfg.model.n_class = num_classes

    train_dataset, train_loader, val_loader, test_loader = _build_loaders(dataset, shuffle, cfg.train, seed=seed)

    sample_tensor, _ = train_dataset[0]
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
    labels = train_dataset.get_class_labels()
    positives = labels.sum(dim=0)
    total = torch.tensor(labels.shape[0], dtype=positives.dtype, device=positives.device)
    negatives = total - positives
    with torch.no_grad():
        pos_weight = torch.where(positives > 0, negatives / (positives + 1e-8), torch.zeros_like(positives))
    if pos_weight.numel() > 0:
        pos_weight = pos_weight.to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters(), lr=cfg.train.learning_rate)

    best_state = None
    best_metric = 0
    best_epoch = 0

    log_every = max(1, cfg.train.log_interval)

    log.info("Starting training for %s epochs...", cfg.train.num_epochs)
    for epoch in range(1, cfg.train.num_epochs + 1):
        train_loss, train_stats = _train_epoch(model, train_loader, criterion, optimizer, device)

        if val_loader is not None:
            val_loss, val_stats, _, _ = _evaluate(model, val_loader, criterion, device)
            if cfg.train.save_best and val_stats["accuracy"] > best_metric:
                best_metric = val_stats["accuracy"]
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
            if epoch % log_every == 0 or epoch == 1 or epoch == cfg.train.num_epochs:
                log.info(
                    "Epoch %03d | Train stats:  Loss=%.4f Acc=%.4f F1=%.4f Precision=%.4f Recall=%.4f",
                    epoch,
                    train_loss,
                    train_stats["accuracy"],
                    train_stats["f1"],
                    train_stats["precision"],
                    train_stats["recall"]
                )
                log.info(
                    "          | Val stats:    Loss=%.4f Acc=%.4f F1=%.4f Precision=%.4f Recall=%.4f",
                    val_loss,
                    val_stats["accuracy"],
                    val_stats["f1"],
                    val_stats["precision"],
                    val_stats["recall"]
                )
                log.info(
                    "-----------------------------------------------------------------------------------------"
                )
        else:
            if epoch % log_every == 0 or epoch == 1 or epoch == cfg.train.num_epochs:
                log.info(
                    "Epoch %03d | Train stats:  Loss=%.4f Acc=%.4f F1=%.4f Precision=%.4f Recall=%.4f",
                    epoch,
                    train_loss,
                    train_stats["accuracy"],
                    train_stats["f1"],
                    train_stats["precision"],
                    train_stats["recall"]
                )

    if best_state is not None:
        log.info("Loading best model from epoch %d with val_acc=%.4f", best_epoch, best_metric)
        model.load_state_dict(best_state)

    test_loss, test_stats, test_logits, test_targets = _evaluate(model, test_loader, criterion, device)
    log.info(
        "Test stats | loss=%.4f acc=%.4f f1=%.4f precision=%.4f recall=%.4f",
        test_loss,
        test_stats["accuracy"],
        test_stats["f1"],
        test_stats["precision"],
        test_stats["recall"],
    )

    pr_curves = _precision_recall_analysis(test_logits, test_targets)
    test_stats = _multilabel_stats(test_logits, test_targets, per_class=True)

    media_dir = Path(get_original_cwd()) / "media"
    media_dir.mkdir(parents=True, exist_ok=True)
    plt.switch_backend("Agg")
    pr_plot_path = _plot_precision_recall_curves(pr_curves, media_dir, cfg.model.name, cfg.dataset.name)
    log.info("Saved precision-recall curves to %s", pr_plot_path)

    pr_scatter_path = _plot_per_class_precision_recall_scatter(
        test_stats["precision"],
        test_stats["recall"],
        media_dir,
        cfg.model.name,
        cfg.dataset.name,
    )
    log.info("Saved per-class precision/recall scatter plot to %s", pr_scatter_path)

    checkpoint_path = _resolve_checkpoint_path(cfg.model, cfg.dataset.name)
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)
    log.info("Saved checkpoint to %s", checkpoint_path)

if __name__ == "__main__":
    run()
