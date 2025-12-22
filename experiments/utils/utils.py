import os
import torch
import numpy as np
import pickle
from omegaconf import DictConfig, OmegaConf
from typing import Dict, List, Tuple, Optional
from crp.attribution import CondAttribution
from zennit.composites import EpsilonPlusFlat
from pathlib import Path
from datasets import get_dataset
from models import get_fn_model_loader, get_canonizer
from utils.visualizations import plot_training_loss, plot_metrics_over_time, plot_cosine_similarity, plot_auc_before_after, plot_uniqueness_before_after, visualize_confusion_trajectories
from utils.metrics import get_accuracy, get_avg_precision, get_uniqueness, compute_auc_performance, get_auconf, get_confusion_matrices
from utils.sim_matrix import reorder_similarity_matrix

def get_save_dir(cfg: DictConfig) -> Path:
    cache_dir = Path(cfg.experiment.out)
    model_name = f"alpha{cfg.cav.alpha}"
    if cfg.cav.beta is not None:
        model_name += f"_beta{cfg.cav.beta}_n_targets{cfg.cav.n_targets}"
    model_name += f"_lr{cfg.train.learning_rate}"
    if cfg.cav.optimal_init:
        model_name += "_opt"
    return cache_dir / model_name

def initialize_weights(C: torch.Tensor, labels: torch.Tensor, alpha: float, beta: float, n_targets: int, device: torch.device) -> torch.Tensor:
    """Initialize the weights for the orthogonality loss.
    Args:
        C (torch.Tensor): Similarity matrix of shape (n_concepts, n_concepts).
        labels (torch.Tensor): Binary labels of shape (n_samples, n_concepts).
        alpha (float): Weight for the L2 regularization term.
        beta (float): Weight for the orthogonality term.
        n_targets (int): Number of target pairs to consider for orthogonality.
        device (torch.device): Device to perform computations on.
    Returns:
        torch.Tensor: Weights matrix of shape (n_concepts, n_concepts).
    """
    if alpha == 0:
        return 0
    
    weights = alpha * torch.ones_like(C, device=device)

    if beta != None:
        # Extract the rows and cols
        similarities = torch.triu(C.abs(), diagonal=1).clone()
        sorted_indices = torch.argsort(similarities.flatten(), descending=True)
        rows = sorted_indices // similarities.size(1)
        cols = sorted_indices % similarities.size(1)
        
        # Iterate through ordered pairs and select the first k valid ones
        selected_pairs = []
        for i, j in zip(rows.tolist(), cols.tolist()):
            if len(set(torch.where(labels[:, i] == 1)[0].tolist()).intersection(
                   set(torch.where(labels[:, j] == 1)[0].tolist()))) != 0:
                selected_pairs.append((i, j))
                
            if len(selected_pairs) == n_targets:
                break
        
        # Assign weights for the selected pairs
        for i, j in selected_pairs:
            weights[i, j] = np.sqrt(beta)
            weights[j, i] = np.sqrt(beta)

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