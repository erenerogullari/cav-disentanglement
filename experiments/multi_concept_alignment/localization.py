import hashlib
import json
import logging
import math
import pickle
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from crp.attribution import CondAttribution
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader
from zennit.composites import EpsilonPlusFlat

from datasets import get_dataset
from datasets.celeba.celeba_attacked import ARTIFACT_CONCEPT_POOL
from experiments.model_correction.evaluate_heatmaps import _dilate_binary_masks
from experiments.model_correction.utils import load_base_model
from models import get_canonizer
from utils.localization import (
    binarize_heatmaps,
    compute_concept_relevance,
    get_localizations,
)

log = logging.getLogger(__name__)

LOCALIZATION_COLUMNS = [
    "concept_relevance",
    "otsu_iou",
    "intersection_over_true_mask",
    "mask_area_fraction",
]
LOCALIZATION_PER_SAMPLE_COLUMNS = [
    "variant_id",
    "num_concepts",
    "concept_set",
    "cav_method",
    "alpha",
    "cav_variant",
    "concept",
    "mask_variant",
    "sample_id",
    *[f"label_{concept}" for concept in ARTIFACT_CONCEPT_POOL],
    *LOCALIZATION_COLUMNS,
]


def select_localization_sample_ids(
    dataset: Any,
    concept: str,
    max_samples: int,
    seed: int,
) -> np.ndarray:
    if concept == "brightness":
        return np.array([], dtype=int)
    positives = np.asarray(dataset.sample_ids_by_artifact.get(concept, []), dtype=int)
    test_ids = np.asarray(dataset.idxs_test, dtype=int)
    candidates = np.intersect1d(positives, test_ids)
    if len(candidates) <= max_samples:
        return candidates

    concept_seed = int.from_bytes(
        hashlib.sha256(concept.encode("utf-8")).digest()[:4], "big"
    )
    rng = np.random.default_rng((int(seed) + concept_seed) % (2**32))
    return np.sort(rng.choice(candidates, size=max_samples, replace=False))


def compute_localization_metrics(
    heatmaps: torch.Tensor, masks: torch.Tensor
) -> dict[str, torch.Tensor]:
    if heatmaps.shape != masks.shape or heatmaps.ndim != 3:
        raise ValueError(
            "Expected heatmaps and masks with matching (N, H, W) shapes, got "
            f"{tuple(heatmaps.shape)} and {tuple(masks.shape)}."
        )
    heatmaps = heatmaps.detach().cpu()
    masks = masks.detach().cpu().bool()
    binary_heatmaps = binarize_heatmaps(heatmaps, thresholding="otsu").bool()
    intersection = torch.logical_and(binary_heatmaps, masks).sum((1, 2)).float()
    union = torch.logical_or(binary_heatmaps, masks).sum((1, 2)).float()
    true_area = masks.sum((1, 2)).float()
    image_area = masks.shape[-2] * masks.shape[-1]
    return {
        "concept_relevance": compute_concept_relevance(
            heatmaps, masks.float()
        ).float(),
        "otsu_iou": intersection / (union + 1e-10),
        "intersection_over_true_mask": intersection / (true_area + 1e-10),
        "mask_area_fraction": true_area / image_area,
    }


def summarize_localization_rows(df: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        column
        for column in [
            "variant_id",
            "num_concepts",
            "concept_set",
            "cav_method",
            "alpha",
            "cav_variant",
            "concept",
            "mask_variant",
        ]
        if column in df.columns
    ]
    rows = []
    for keys, group in df.groupby(group_columns, sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_columns, keys))
        row["n"] = len(group)
        for metric in LOCALIZATION_COLUMNS:
            values = group[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=0))
            row[f"{metric}_sem"] = float(values.std(ddof=0) / math.sqrt(len(values)))
        rows.append(row)
    return pd.DataFrame(rows)


def _plot_localization_summary(df: pd.DataFrame, output_dir: Path) -> None:
    if df.empty:
        return
    metric_labels = {
        "concept_relevance": "Concept relevance",
        "otsu_iou": "Otsu IoU",
        "intersection_over_true_mask": "Intersection / true mask",
        "mask_area_fraction": "Mask area fraction",
    }
    for (concept, mask_variant), group in df.groupby(
        ["concept", "mask_variant"], sort=False
    ):
        variants = list(group["cav_variant"])
        fig, axes = plt.subplots(1, len(metric_labels), figsize=(12, 3))
        for axis, (metric, label) in zip(axes, metric_labels.items()):
            means = group[f"{metric}_mean"].to_numpy(dtype=float)
            sems = group[f"{metric}_sem"].to_numpy(dtype=float)
            axis.bar(variants, means, yerr=sems, capsize=3)
            axis.set_title(label)
            axis.set_ylim(0, max(1.0, float(np.nanmax(means + sems)) * 1.05))
            axis.tick_params(axis="x", rotation=20)
        fig.suptitle(f"{concept} ({mask_variant})")
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            fig.savefig(
                output_dir / f"localization_{concept}_{mask_variant}.{suffix}",
                bbox_inches="tight",
            )
        plt.close(fig)


def _instantiate_heatmap_dataset(cfg: DictConfig):
    dataset_kwargs = OmegaConf.to_container(cfg.dataset, resolve=True)
    assert isinstance(dataset_kwargs, dict)
    dataset_name = str(dataset_kwargs.pop("name"))
    dataset_kwargs.pop("_target_", None)
    dataset_kwargs.pop("cache_namespace", None)
    dataset_kwargs.pop("shuffle", None)
    return get_dataset(dataset_name + "_hm")(**dataset_kwargs)


def _plot_qualitative_grid(
    dataset,
    concept: str,
    images: torch.Tensor,
    heatmaps_by_variant: dict[str, torch.Tensor],
    masks: torch.Tensor,
    output_dir: Path,
    max_images: int,
) -> None:
    nrows = min(max_images, len(images))
    if nrows == 0:
        return
    variants = list(heatmaps_by_variant)
    fig, axes = plt.subplots(
        nrows,
        2 + len(variants),
        figsize=((2 + len(variants)) * 2, nrows * 2),
        squeeze=False,
    )
    for row in range(nrows):
        image = dataset.reverse_normalization(images[row]).permute(1, 2, 0).numpy()
        axes[row, 0].imshow(np.clip(image, 0, 255).astype(np.uint8))
        axes[row, 1].imshow(masks[row].numpy(), cmap="gray", vmin=0, vmax=1)
        for column, variant in enumerate(variants, start=2):
            axes[row, column].imshow(
                heatmaps_by_variant[variant][row].numpy(), cmap="inferno"
            )
    axes[0, 0].set_title("Input")
    axes[0, 1].set_title("Mask")
    for column, variant in enumerate(variants, start=2):
        axes[0, column].set_title(variant)
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle(concept)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"qualitative_{concept}.{suffix}", bbox_inches="tight")
    plt.close(fig)


def evaluate_multi_concept_localization(
    cfg: DictConfig,
    cav_model: nn.Module,
    base_cav_model: nn.Module,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg.train.device)
    dataset = _instantiate_heatmap_dataset(cfg)
    classification_model = load_base_model(cfg, dataset.num_classes, device)
    classification_model.eval()
    attribution = CondAttribution(classification_model)
    composite = EpsilonPlusFlat(canonizers=get_canonizer(cfg.model.name))

    concept_names = dataset.get_concept_names()
    cavs_baseline, _ = base_cav_model.get_params(normalize=True)  # type: ignore
    cavs_orthogonal, _ = cav_model.get_params(normalize=True)  # type: ignore
    cav_sets = {
        "Baseline": cavs_baseline.cpu(),
        "Orthogonal": cavs_orthogonal.cpu(),
    }
    active_concepts = list(dataset.artifact_concepts)
    localizable_concepts = [
        concept
        for concept in active_concepts
        if concept not in set(cfg.localization.exclude_concepts)
    ]
    run_metadata = {
        "variant_id": str(cfg.experiment.variant_id),
        "num_concepts": len(active_concepts),
        "concept_set": json.dumps(active_concepts),
        "cav_method": str(cfg.cav.name),
        "alpha": float(cfg.cav.alpha),
    }

    per_sample_rows: list[dict[str, Any]] = []
    for concept in localizable_concepts:
        sample_ids = select_localization_sample_ids(
            dataset,
            concept,
            max_samples=int(cfg.localization.max_samples_per_concept),
            seed=int(cfg.localization.seed),
        )
        if len(sample_ids) == 0:
            log.warning("No positive test samples available for %s; skipping.", concept)
            continue

        subset = dataset.get_subset_by_idxs(sample_ids.tolist())
        dataloader = DataLoader(
            subset,
            batch_size=cfg.train.batch_size,
            num_workers=cfg.train.num_workers,
            shuffle=False,
        )
        concept_id = concept_names.index(concept)
        images_by_batch = []
        masks_by_batch = []
        heatmaps_by_variant: dict[str, list[torch.Tensor]] = {
            variant: [] for variant in cav_sets
        }
        for batch in dataloader:
            images, _, artifact_masks = batch
            images_by_batch.append(images.detach().cpu())
            masks_by_batch.append(artifact_masks[concept].detach().cpu())
            for variant, cavs in cav_sets.items():
                _, heatmaps = get_localizations(
                    images.clone(),
                    cavs[concept_id],
                    attribution,
                    composite,
                    {"layer_name": cfg.cav.layer},
                    device,
                    model_name=cfg.model.name,
                    model=classification_model,
                )
                heatmaps_by_variant[variant].append(heatmaps)

        images = torch.cat(images_by_batch)
        raw_masks = torch.cat(masks_by_batch)
        heatmaps = {
            variant: torch.cat(parts)
            for variant, parts in heatmaps_by_variant.items()
        }
        mask_variants = {"raw": raw_masks}
        if concept in {"box", "watermark"}:
            mask_variants["dilated_3px"] = _dilate_binary_masks(raw_masks, 3)

        for cav_variant, concept_heatmaps in heatmaps.items():
            for mask_variant, masks in mask_variants.items():
                metric_values = compute_localization_metrics(concept_heatmaps, masks)
                for local_index, sample_id in enumerate(sample_ids):
                    row = {
                        **run_metadata,
                        "cav_variant": cav_variant,
                        "concept": concept,
                        "mask_variant": mask_variant,
                        "sample_id": int(sample_id),
                    }
                    for artifact in ARTIFACT_CONCEPT_POOL:
                        labels = dataset.artifact_labels_by_concept.get(artifact)
                        row[f"label_{artifact}"] = (
                            int(labels[sample_id]) if labels is not None else 0
                        )
                    for metric, values in metric_values.items():
                        row[metric] = float(values[local_index])
                    per_sample_rows.append(row)

        _plot_qualitative_grid(
            dataset,
            concept,
            images,
            heatmaps,
            raw_masks,
            output_dir,
            max_images=int(cfg.localization.qualitative_samples),
        )

    df_per_sample = pd.DataFrame(
        per_sample_rows, columns=LOCALIZATION_PER_SAMPLE_COLUMNS
    )
    if df_per_sample.empty:
        df_summary = pd.DataFrame()
    else:
        df_summary = summarize_localization_rows(df_per_sample)
    _plot_localization_summary(df_summary, output_dir)

    compatibility_metrics: dict[str, float] = {}
    compatibility_metric_names = {
        "concept_relevance": "concept_rel",
        "otsu_iou": "iou",
        "intersection_over_true_mask": "intersection_over_true_mask",
    }
    for _, row in df_summary.iterrows():
        concept_name = str(row["concept"])
        if row["mask_variant"] == "dilated_3px":
            concept_name += "_dilated_3px"
        for metric, legacy_metric in compatibility_metric_names.items():
            key = f"{legacy_metric}_{concept_name}_{row['cav_variant']}"
            compatibility_metrics[key] = float(row[f"{metric}_mean"])
            compatibility_metrics[f"{key}_sem"] = float(row[f"{metric}_sem"])

    df_per_sample.to_csv(output_dir / "localization_per_sample.csv", index=False)
    df_summary.to_csv(output_dir / "localization_summary.csv", index=False)
    with open(output_dir / "concept_relevance.pkl", "wb") as file:
        pickle.dump(compatibility_metrics, file)
    return df_summary, df_per_sample
