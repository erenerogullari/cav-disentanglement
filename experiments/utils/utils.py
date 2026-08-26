import os
import torch
import numpy as np
import pickle
from omegaconf import DictConfig, OmegaConf
from typing import Any, Dict, List, Sequence
from crp.attribution import CondAttribution
from zennit.composites import EpsilonPlusFlat
from pathlib import Path
from datasets import get_dataset
from models import get_fn_model_loader, get_canonizer
from utils.visualizations import plot_training_loss, plot_metrics_over_time, plot_cosine_similarity, plot_auc_before_after, plot_uniqueness_before_after, visualize_confusion_trajectories
from utils.metrics import get_accuracy, get_avg_precision, get_uniqueness, compute_auc_performance, get_auconf, get_confusion_matrices
from utils.sim_matrix import reorder_similarity_matrix


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _sanitize_path_component(value: Any) -> str:
    text = str(value)
    sanitized = "".join(char if char.isalnum() or char in "._-" else "-" for char in text)
    sanitized = "-".join(part for part in sanitized.split("-") if part)
    return sanitized or "none"


def get_target_concepts(cav_cfg: DictConfig | Dict[str, Any]) -> list[str]:
    """Return target concept names from a CAV-like config as plain strings."""
    raw_targets = cav_cfg.get("target_concepts", [])
    return [str(concept) for concept in _as_list(raw_targets)]


def get_dataset_cache_namespace(dataset_cfg: DictConfig | Dict[str, Any]) -> str:
    """Return an optional variant-aware namespace for activation/CAV caches."""
    return str(dataset_cfg.get("cache_namespace", dataset_cfg.get("name")))


def format_orthogonality_config_name(
    alpha: Any,
    beta: Any = None,
    target_concepts: Sequence[str] | None = None,
) -> str:
    """Stable directory suffix for the orthogonality weighting configuration."""
    targets = [str(concept) for concept in _as_list(target_concepts)]
    parts = [f"alpha{_sanitize_path_component(alpha)}"]
    if targets:
        parts.append(f"beta{_sanitize_path_component(beta)}")
        parts.append(
            "targets-" + "-".join(_sanitize_path_component(concept) for concept in targets)
        )
    return "_".join(parts)


def get_save_dir(cfg: DictConfig) -> Path:
    cache_dir = Path(cfg.experiment.out)
    target_concepts = get_target_concepts(cfg.cav)
    model_name = format_orthogonality_config_name(
        cfg.cav.alpha,
        cfg.cav.get("beta", None),
        target_concepts,
    )
    model_name += f"_lr{cfg.train.learning_rate}"
    if cfg.cav.optimal_init:
        model_name += "_opt"
    return cache_dir / model_name


def initialize_weights(
    C: torch.Tensor,
    concept_names: Sequence[str],
    alpha: float,
    beta: float | None,
    target_concepts: Sequence[str] | None,
    device: torch.device,
) -> torch.Tensor:
    """Initialize the weights for the orthogonality loss.

    Args:
        C (torch.Tensor): Similarity matrix of shape (n_concepts, n_concepts).
        concept_names (Sequence[str]): Concept names in the same order as C.
        alpha (float): Weight for pairs without any target concept.
        beta (float | None): Weight for pairs involving at least one target concept.
        target_concepts (Sequence[str] | None): Concepts that should use beta.
        device (torch.device): Device to perform computations on.

    Returns:
        torch.Tensor: Weights matrix of shape (n_concepts, n_concepts).
    """
    names = [str(concept) for concept in concept_names]
    if C.shape != (len(names), len(names)):
        raise ValueError(
            f"Expected C shape {(len(names), len(names))} for {len(names)} concepts, "
            f"got {tuple(C.shape)}."
        )

    targets = [str(concept) for concept in _as_list(target_concepts)]
    missing_targets = sorted(set(targets) - set(names))
    if missing_targets:
        raise ValueError(
            "Unknown target_concepts: "
            f"{missing_targets}. Available concepts: {names}."
        )

    weights = torch.full_like(C, float(alpha), device=device)
    if not targets:
        return weights

    if beta is None:
        raise ValueError("cav.beta must be set when cav.target_concepts is non-empty.")

    target_indices = torch.tensor(
        [names.index(concept) for concept in targets],
        device=device,
        dtype=torch.long,
    )
    target_mask = torch.zeros(C.shape[0], dtype=torch.bool, device=device)
    target_mask[target_indices] = True
    pair_mask = target_mask[:, None] | target_mask[None, :]
    weights[pair_mask] = float(beta)
    return weights


def save_results(cavs: torch.Tensor, metrics: Dict, save_dir: Path) -> None:
    """Save the CAVs and metrics to the specified directory."""
    cavs_normalized = cavs / torch.norm(cavs, dim=1, keepdim=True)
    torch.save(cavs_normalized, f'{save_dir}/cavs.pt')

    with open(f"{str(save_dir)}/metrics/auc_hist.pkl", "wb") as f:
        pickle.dump(metrics['auc_hist'], f)

    with open(f"{str(save_dir)}/metrics/uniqueness_hist.pkl", "wb") as f:
        pickle.dump(metrics['uniqueness_hist'], f)

    with open(f"{str(save_dir)}/metrics/precision_hist.pkl", "wb") as f:
        pickle.dump(metrics['precision_hist'], f)

    with open(f"{str(save_dir)}/metrics/confusion_matrix_hist.pkl", "wb") as f:
        pickle.dump(metrics['confusion_matrix_hist'], f)

    with open(f"{str(save_dir)}/metrics/cav_loss_hist.pkl", "wb") as f:
        pickle.dump(metrics['cav_loss_hist'], f)

    with open(f"{str(save_dir)}/metrics/orth_loss_hist.pkl", "wb") as f:
        pickle.dump(metrics['orth_loss_hist'], f)


def save_plots(cavs: torch.Tensor, cavs_original: torch.Tensor, metrics: Dict, x_latent: torch.Tensor, labels: torch.Tensor, concepts: List, save_dir: Path) -> None:
    """Generate and save plots for the experiment."""
    os.makedirs(f"{str(save_dir)}/media", exist_ok=True)

    plot_training_loss( 
        cav_loss_history=metrics['cav_loss_hist'], 
        orthogonality_loss_history=metrics['orth_loss_hist'], 
        save_path=f"{str(save_dir)}/media/training_loss.png"
    )
    
    cav_performance_history = np.mean(np.array(metrics['auc_hist']), axis=1)
    cav_uniqueness_history = np.mean(np.array(metrics['uniqueness_hist']), axis=1)
    epochs_logged = [10*i for i in range(len(cav_performance_history))]
    plot_metrics_over_time(
        epochs_logged=epochs_logged,
        cav_performance_history=cav_performance_history,
        avg_precision_hist=metrics['precision_hist'],
        cav_uniqueness_history=cav_uniqueness_history,
        threshold=None,
        early_exit_epoch=metrics['early_exit_epoch'],
        save_path=f"{str(save_dir)}/media/metrics_plot.png"
    )

    cavs_normalized = cavs / torch.norm(cavs, dim=1, keepdim=True)
    cos_sim_matrix = cavs_normalized @ cavs_normalized.T
    cavs_original = cavs_original.detach().cpu()
    cos_sim_matrix_original = cavs_original @ cavs_original.T 
    plot_cosine_similarity(
        cos_sim_matrix_original=cos_sim_matrix_original,
        cos_sim_matrix=cos_sim_matrix,
        concepts=concepts,  # List of concept names
        save_path=f"{str(save_dir)}/media/cos_sim_before_after.png"
    )

    auc_before = compute_auc_performance(cavs_original, x_latent, labels)
    auc_after = compute_auc_performance(cavs_normalized, x_latent, labels)
    auc_diff = np.array(auc_after) - np.array(auc_before)
    sorted_indices_auc = np.argsort(auc_diff)
    sorted_concepts_auc = [concepts[i] for i in sorted_indices_auc]
    sorted_auc_before = [auc_before[i] for i in sorted_indices_auc]
    sorted_auc_after = [auc_after[i] for i in sorted_indices_auc]
    plot_auc_before_after(
        auc_before=sorted_auc_before,
        auc_after=sorted_auc_after,
        concepts=sorted_concepts_auc,  # List of concept names
        save_path=f"{str(save_dir)}/media/auc_before_after.png"
    )

    uniqueness_before = get_uniqueness(cos_sim_matrix_original)
    uniqueness_after = get_uniqueness(cos_sim_matrix)

    unq_diff = np.array(uniqueness_after) - np.array(uniqueness_before)
    sorted_indices_unq = np.argsort(unq_diff)
    sorted_concepts_unq = [concepts[i] for i in sorted_indices_unq]
    sorted_uniqueness_before = [uniqueness_before[i] for i in sorted_indices_unq]
    sorted_uniqueness_after = [uniqueness_after[i] for i in sorted_indices_unq]

    plot_uniqueness_before_after(
        uniqueness_before=sorted_uniqueness_before,
        uniqueness_after=sorted_uniqueness_after,
        concepts=sorted_concepts_unq, 
        save_path=f"{str(save_dir)}/media/uniqueness_before_after.png"
    )

    visualize_confusion_trajectories(metrics['confusion_matrix_hist'], save_path=f"{save_dir}/media/confusion_trajectories.png")
