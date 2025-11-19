import random
from pathlib import Path
from typing import Optional, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.cav import compute_cavs
from utils.metrics import compute_auc_performance, get_uniqueness
from utils.visualizations import (
    plot_auc_before_after,
    plot_cosine_similarity,
    plot_uniqueness_before_after,
)

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


def resolve_concept_names(config: DictConfig, n_concepts: int) -> List[str]:
    """Fetch concept names from the dataset, falling back to generic labels."""
    try:
        dataset = instantiate(config.dataset)
        concept_names = dataset.get_concept_names()  # type: ignore[attr-defined]
    except Exception as exc:  # pragma: no cover - defensive fallback
        log.warning("Failed to load dataset for concept names (%s). Falling back to generic names.", exc)
        concept_names = None
    finally:
        dataset = None  # type: ignore[assignment]

    if not concept_names:
        concept_names = [f"concept_{idx}" for idx in range(n_concepts)]
    return concept_names


def plot_auc_uniqueness_over_epochs(epochs: List[int], auc_history: List[float], uniqueness_history: List[float], save_path: Path) -> None:
    if not epochs:
        return
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, auc_history, marker="o", label="Mean AUC")
    plt.plot(epochs, uniqueness_history, marker="o", label="Mean Uniqueness")
    plt.xlabel("Epochs")
    plt.ylabel("Score")
    plt.title("AUC & Uniqueness over Training")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def find_cached_dir_models(
    config: DictConfig,
    alphas: List[float],
    cache_dir: Path,
    n_concepts: int,
    n_features: int,
) -> Dict[float, nn.Module]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    dir_models: Dict[float, nn.Module] = {}
    cav_cfg = {"_target_": config.dir_model["_target_"]}

    for alpha in alphas:
        alpha_dir = cache_dir / f"alpha{alpha}"
        state_path = alpha_dir / "state_dict.pth"
        if not state_path.exists():
            continue
        log.info("Found cached direction model for alpha=%s in %s.", alpha, state_path)
        dir_model = instantiate(
            cav_cfg,
            n_concepts=n_concepts,
            n_features=n_features,
            device="cpu",
        )
        state_dict = torch.load(state_path, map_location="cpu")
        dir_model.load_state_dict(state_dict)
        dir_model.eval()
        dir_models[alpha] = dir_model

    return dir_models


def build_dataloader(encodings, labels, batch_size: int) -> DataLoader:
    dataset = torch.utils.data.TensorDataset(encodings, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def train_dir_model(config: DictConfig, encodings: torch.Tensor, labels: torch.Tensor) -> Dict[float, nn.Module]:

    alphas = config.dir_model.alphas
    cache_dir = Path("results") / "diffae" / "dir_models" / str(config.dir_model.name)

    log.info("Seeding RNGs with %s", config.experiment.seed)
    seed_everything(int(config.experiment.seed))

    device = torch.device(config.experiment.device)
    log.info(f"Using device {device}")

    log.info('Initializing dataloader')
    encodings = encodings.float()
    labels = labels.float()
    dataloader = build_dataloader(encodings, labels, config.experiment.batch_size)

    n_concepts = labels.shape[1]
    n_features = encodings.shape[1]

    dir_models = find_cached_dir_models(config, alphas, cache_dir, n_concepts, n_features)
    alphas_to_train = [alpha for alpha in alphas if alpha not in dir_models.keys()]

    if len(alphas_to_train) == 0:
        log.info("All CAVs found in cache. Skipping training.")
        return dir_models

    concept_names = resolve_concept_names(config, n_concepts)
    cavs_original_raw, _ = compute_cavs(encodings, labels, type=config.dir_model.name, normalize=True)
    cavs_original = cavs_original_raw.detach().cpu()
    cavs_original_normalized = torch.nn.functional.normalize(cavs_original, dim=1)
    auc_before = compute_auc_performance(cavs_original_normalized, encodings, labels)
    uniqueness_before = get_uniqueness(cavs_original_normalized)
    cos_sim_matrix_original = cavs_original_normalized @ cavs_original_normalized.T

    metrics_interval = getattr(config.dir_model, "metrics_interval", 10)

    for alpha in alphas_to_train:
        log.info(f"Training direction model for alpha={alpha}")

        log.info('Building components')
        cav_cfg = {"_target_": config.dir_model["_target_"]}
        dir_model = instantiate(
            cav_cfg,
            n_concepts=n_concepts,
            n_features=n_features,
            device=device,
        )
        dir_model = dir_model.to(device)
        optimizer = optim.Adam(dir_model.parameters(), lr=config.dir_model.learning_rate)

        epochs_logged: List[int] = []
        auc_history: List[float] = []
        uniqueness_history: List[float] = []

        for epoch in tqdm(range(config.dir_model.n_epochs), desc="Epochs"):
            dir_model.train()
            total_cav_loss = 0.0
            total_orth_loss = 0.0
            num_batches = 0

            for batch_encs, batch_labels in dataloader:
                batch_encs = batch_encs.to(device=device, dtype=torch.float32)
                batch_labels = batch_labels.to(device=device, dtype=torch.float32).clamp(min=0)

                optimizer.zero_grad()
                cav_loss, orth_loss = dir_model.train_step(batch_encs, batch_labels, alpha)
                loss = cav_loss + orth_loss
                loss.backward()
                optimizer.step()

                total_cav_loss += cav_loss.item()
                total_orth_loss += orth_loss.item()
                num_batches += 1

            if (epoch + 1) % metrics_interval == 0 or epoch == config.dir_model.n_epochs - 1:
                cavs_current, _ = dir_model.get_params()  # type: ignore[assignment]
                cavs_current = cavs_current.detach().cpu()
                cavs_normalized = torch.nn.functional.normalize(cavs_current, dim=1)
                auc_scores = compute_auc_performance(cavs_normalized, encodings, labels)
                mean_auc = float(np.nanmean(auc_scores))
                uniqueness_scores = get_uniqueness(cavs_normalized)
                mean_uniqueness = float(np.mean(uniqueness_scores))
                epochs_logged.append(epoch + 1)
                auc_history.append(mean_auc)
                uniqueness_history.append(mean_uniqueness)

        dir_model_cpu = dir_model.to("cpu")
        dir_model_cpu.eval()

        alpha_dir = cache_dir / f"alpha{alpha}"
        alpha_dir.mkdir(parents=True, exist_ok=True)
        torch.save(dir_model_cpu.state_dict(), alpha_dir / "state_dict.pth")
        dir_models[alpha] = dir_model_cpu

        cavs_trained, _ = dir_model_cpu.get_params()  # type: ignore[assignment]
        cavs_trained = cavs_trained.detach().cpu()
        cavs_trained_normalized = torch.nn.functional.normalize(cavs_trained, dim=1)
        auc_after = compute_auc_performance(cavs_trained_normalized, encodings, labels)
        uniqueness_after = get_uniqueness(cavs_trained_normalized)
        cos_sim_matrix_after = cavs_trained_normalized @ cavs_trained_normalized.T

        plot_auc_uniqueness_over_epochs(
            epochs_logged,
            auc_history,
            uniqueness_history,
            alpha_dir / "auc_vs_uniqueness.png",
        )
        plot_cosine_similarity(
            cos_sim_matrix_original=cos_sim_matrix_original,
            cos_sim_matrix=cos_sim_matrix_after,
            concepts=concept_names,
            save_path=str(alpha_dir / "cos_sim_before_after.png"),
        )
        plot_auc_before_after(
            auc_before=auc_before.tolist(),
            auc_after=auc_after.tolist(),
            concepts=concept_names,
            save_path=str(alpha_dir / "auc_before_after.png"),
        )
        plot_uniqueness_before_after(
            uniqueness_before=uniqueness_before.tolist(),
            uniqueness_after=uniqueness_after.tolist(),
            concepts=concept_names,
            save_path=str(alpha_dir / "uniqueness_before_after.png"),
        )

    return dir_models
