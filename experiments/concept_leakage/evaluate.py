import logging
from pathlib import Path
from typing import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

from experiments.utils.localization import get_localization
from models import get_canonizer
from utils.localization import binarize_heatmaps, compute_concept_relevance

log = logging.getLogger(__name__)


def compute_localization_metrics(
    heatmap: torch.Tensor,
    target_mask: torch.Tensor,
    eps: float = 1e-10,
) -> dict[str, float | int]:
    if heatmap.ndim != 2 or target_mask.shape != heatmap.shape:
        raise ValueError(
            "Expected matching two-dimensional heatmap and mask, got "
            f"{tuple(heatmap.shape)} and {tuple(target_mask.shape)}."
        )

    positive = heatmap.detach().cpu().clamp(min=0)
    target = target_mask.detach().cpu().bool()
    positive_total = float(positive.sum())
    zero_heatmap = int(positive_total <= eps)
    target_relevance = float(compute_concept_relevance(positive, target.float()))

    if zero_heatmap:
        pointing = 0
        predicted = torch.zeros_like(target)
    else:
        max_index = int(positive.flatten().argmax())
        pointing = int(target.flatten()[max_index])
        min_side = min(positive.shape)
        kernel_size = min(7, min_side if min_side % 2 == 1 else min_side - 1)
        kernel_size = max(kernel_size, 1)
        predicted = binarize_heatmaps(
            positive.unsqueeze(0), kernel_size=kernel_size
        )[0].bool()

    intersection = float(torch.logical_and(predicted, target).sum())
    union = float(torch.logical_or(predicted, target).sum())
    target_area = float(target.sum())
    return {
        "target_relevance": target_relevance,
        "pointing_game": pointing,
        "iou": intersection / (union + eps),
        "intersection_over_gt": intersection / (target_area + eps),
        "mask_area": target_area,
        "positive_relevance_total": positive_total,
        "zero_heatmap": zero_heatmap,
    }


def compute_directed_leakage(
    heatmap: torch.Tensor,
    source_mask: torch.Tensor,
    distractor_mask: torch.Tensor,
    eps: float = 1e-10,
) -> tuple[float, int]:
    if (
        heatmap.ndim != 2
        or source_mask.shape != heatmap.shape
        or distractor_mask.shape != heatmap.shape
    ):
        raise ValueError("Heatmap and both concept masks must have matching 2-D shapes.")
    positive = heatmap.detach().cpu().clamp(min=0)
    exclusive_distractor = (
        distractor_mask.detach().cpu().bool()
        & ~source_mask.detach().cpu().bool()
    )
    exclusive_area = int(exclusive_distractor.sum())
    leakage = float(
        (positive * exclusive_distractor.to(positive)).sum()
        / (positive.sum() + eps)
    )
    return leakage, exclusive_area


def _select_positive_indices(
    dataset,
    concept_name: str,
    max_samples: int | None,
    seed: int,
) -> np.ndarray:
    candidates = np.asarray(dataset.sample_ids_by_concept[concept_name], dtype=int)
    if max_samples is not None and len(candidates) > max_samples:
        rng = np.random.default_rng(seed)
        candidates = np.sort(rng.choice(candidates, size=max_samples, replace=False))
    return candidates


def _save_localization_overlay(
    image: torch.Tensor,
    heatmap: torch.Tensor,
    mask: torch.Tensor,
    dataset,
    title: str,
    save_path: Path,
) -> None:
    image_np = (
        dataset.reverse_normalization(image.detach().cpu())
        .permute(1, 2, 0)
        .clamp(0, 255)
        .numpy()
        .astype(np.uint8)
    )
    heatmap_np = heatmap.detach().cpu().clamp(min=0).numpy()
    mask_np = mask.detach().cpu().bool().numpy()

    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    axes[0].imshow(image_np)
    axes[0].set_title("Image")
    axes[1].imshow(heatmap_np, cmap="inferno")
    axes[1].set_title("Positive relevance")
    axes[2].imshow(mask_np, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Ground truth")
    axes[3].imshow(image_np)
    axes[3].imshow(heatmap_np, cmap="inferno", alpha=0.55)
    axes[3].contour(mask_np, levels=[0.5], colors="cyan", linewidths=1)
    axes[3].set_title("Overlay")
    for axis in axes:
        axis.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def _summarize_localization(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "target_relevance",
        "pointing_game",
        "iou",
        "intersection_over_gt",
        "mask_area",
        "positive_relevance_total",
        "zero_heatmap",
    ]
    rows = []
    for concept_name, concept_frame in frame.groupby("concept", sort=False):
        for metric in metrics:
            values = concept_frame[metric].astype(float)
            rows.append(
                {
                    "concept": concept_name,
                    "metric": metric,
                    "count": len(values),
                    "mean": values.mean(),
                    "std": values.std(ddof=1),
                    "sem": values.sem(ddof=1),
                }
            )
    return pd.DataFrame(rows)


def _build_leakage_tables(
    frame: pd.DataFrame, concept_names: Sequence[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    matrix = pd.DataFrame(np.nan, index=concept_names, columns=concept_names)
    counts = pd.DataFrame(0, index=concept_names, columns=concept_names, dtype=int)
    if frame.empty:
        return matrix, counts

    grouped = frame.groupby(["source_concept", "mask_concept"], sort=False)
    for (source_name, mask_name), pair_frame in grouped:
        matrix.loc[source_name, mask_name] = pair_frame["leakage"].mean()
        counts.loc[source_name, mask_name] = len(pair_frame)
    return matrix, counts


def _save_leakage_figure(matrix: pd.DataFrame, save_path: Path) -> None:
    values = matrix.to_numpy(dtype=float)
    fig, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(values, cmap="magma", vmin=0, vmax=1)
    axis.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=40, ha="right")
    axis.set_yticks(range(len(matrix.index)), matrix.index)
    axis.set_xlabel("Ground-truth mask")
    axis.set_ylabel("Source CAV")
    axis.set_title("Directed concept leakage")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            label = "NA" if np.isnan(value) else f"{value:.2f}"
            axis.text(column, row, label, ha="center", va="center", color="white")
    fig.colorbar(image, ax=axis, label="Positive relevance fraction")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def evaluate_concept_leakage(
    cfg: DictConfig,
    model: torch.nn.Module,
    dataset,
    cavs: torch.Tensor,
    save_dir: Path,
    localization_fn: Callable = get_localization,
) -> None:
    concept_names = dataset.get_concept_names()
    if cavs.shape[0] != len(concept_names):
        raise ValueError(
            f"Expected {len(concept_names)} CAVs, got shape {tuple(cavs.shape)}."
        )

    metrics_dir = save_dir / "metrics"
    media_dir = save_dir / "media" / "localization"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    media_dir.mkdir(parents=True, exist_ok=True)

    device = cfg.evaluation.device
    batch_size = int(cfg.evaluation.batch_size)
    max_samples = cfg.evaluation.max_samples_per_concept
    max_samples = None if max_samples is None else int(max_samples)
    overlays_per_concept = int(cfg.evaluation.overlays_per_concept)
    seed = int(cfg.evaluation.random_seed)
    canonizers = get_canonizer(cfg.model.name)

    localization_rows: list[dict] = []
    leakage_rows: list[dict] = []
    labels = dataset.get_labels().bool()

    for source_index, source_name in enumerate(concept_names):
        sample_indices = _select_positive_indices(
            dataset,
            source_name,
            max_samples,
            seed + source_index,
        )
        log.info(
            "Evaluating %s on %s positive validation images.",
            source_name,
            len(sample_indices),
        )
        overlay_count = 0
        for start in range(0, len(sample_indices), batch_size):
            batch_indices = sample_indices[start : start + batch_size]
            images = torch.stack([dataset[int(index)][0] for index in batch_indices])
            masks_by_sample = [
                dataset.get_concept_masks(int(index)) for index in batch_indices
            ]
            heatmaps = localization_fn(
                cavs[source_index].unsqueeze(0),
                images,
                model,
                canonizers,
                cfg.cav.layer,
                cfg.cav.cav_mode,
                device,
                model_name=cfg.model.name,
            )
            if heatmaps.shape != images.shape[:1] + images.shape[-2:]:
                raise ValueError(
                    f"Expected heatmaps shaped {(len(images), *images.shape[-2:])}, "
                    f"got {tuple(heatmaps.shape)}."
                )

            for batch_offset, dataset_index in enumerate(batch_indices):
                dataset_index = int(dataset_index)
                heatmap = heatmaps[batch_offset].detach().cpu()
                masks = masks_by_sample[batch_offset]
                source_mask = masks[source_name]
                image_id = dataset.get_sample_name(dataset_index)
                row = {
                    "concept": source_name,
                    "concept_index": source_index,
                    "dataset_index": dataset_index,
                    "image_id": image_id,
                }
                row.update(compute_localization_metrics(heatmap, source_mask))
                localization_rows.append(row)

                target_relevance = float(row["target_relevance"])
                leakage_rows.append(
                    {
                        "source_concept": source_name,
                        "mask_concept": source_name,
                        "dataset_index": dataset_index,
                        "image_id": image_id,
                        "leakage": target_relevance,
                        "exclusive_mask_area": int(source_mask.sum()),
                    }
                )
                for mask_index, mask_name in enumerate(concept_names):
                    if mask_index == source_index or not labels[dataset_index, mask_index]:
                        continue
                    leakage, exclusive_area = compute_directed_leakage(
                        heatmap, source_mask, masks[mask_name]
                    )
                    leakage_rows.append(
                        {
                            "source_concept": source_name,
                            "mask_concept": mask_name,
                            "dataset_index": dataset_index,
                            "image_id": image_id,
                            "leakage": leakage,
                            "exclusive_mask_area": exclusive_area,
                        }
                    )

                if overlay_count < overlays_per_concept:
                    _save_localization_overlay(
                        images[batch_offset],
                        heatmap,
                        source_mask,
                        dataset,
                        f"{source_name} | COCO image {image_id}",
                        media_dir / f"{source_index}_{source_name.replace(' ', '-')}_{image_id}.png",
                    )
                    overlay_count += 1

    localization_frame = pd.DataFrame(localization_rows)
    leakage_frame = pd.DataFrame(leakage_rows)
    localization_frame.to_csv(metrics_dir / "localization_per_sample.csv", index=False)
    _summarize_localization(localization_frame).to_csv(
        metrics_dir / "localization_summary.csv", index=False
    )
    leakage_frame.to_csv(metrics_dir / "leakage_per_sample.csv", index=False)
    matrix, counts = _build_leakage_tables(leakage_frame, concept_names)
    matrix.to_csv(metrics_dir / "leakage_matrix.csv", index_label="source_concept")
    counts.to_csv(metrics_dir / "leakage_counts.csv", index_label="source_concept")
    _save_leakage_figure(matrix, save_dir / "media" / "leakage_matrix.png")
