import logging
import pickle
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
import tqdm
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader

from crp.attribution import CondAttribution
from crp.image import imgify
from models import get_canonizer, requires_lxt_localization
from zennit.composites import EpsilonPlusFlat

from experiments.model_correction.utils import load_base_model
from experiments.utils.activations import extract_latents
from experiments.utils.utils import get_save_dir
from utils.cav import compute_cavs
from utils.localization import (
    binarize_heatmaps,
    compute_concept_relevance,
    get_localizations,
)
from datasets import get_dataset

log = logging.getLogger(__name__)
CAV_PLOT_ORDER = ["Baseline", "Orthogonal"]
BOX_MASK_DILATION_PIXELS = 3
BOX_DILATED_CONCEPT_NAME = f"box_dilated_{BOX_MASK_DILATION_PIXELS}px"


def _clean_cav(
    cav: torch.Tensor, non_concept_mean: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    if cav.ndim != 1 or non_concept_mean.ndim != 1:
        raise ValueError("CAV and non-concept mean must both be one-dimensional.")
    if cav.shape != non_concept_mean.shape:
        raise ValueError(
            "CAV and non-concept mean must have matching shapes, got "
            f"{tuple(cav.shape)} and {tuple(non_concept_mean.shape)}."
        )

    non_concept_mean = non_concept_mean.to(cav)
    mean_norm_sq = torch.dot(non_concept_mean, non_concept_mean)
    if mean_norm_sq <= eps:
        cleaned = cav
    else:
        cleaned = cav - (
            torch.dot(cav, non_concept_mean) / mean_norm_sq
        ) * non_concept_mean

    cleaned_norm = cleaned.norm()
    if cleaned_norm <= eps:
        raise ValueError("Concept cleaning collapsed the CAV to a zero direction.")
    return cleaned / cleaned_norm


def _clean_cavs(
    cavs: torch.Tensor,
    activations: torch.Tensor,
    labels: torch.Tensor,
    concept_names: list[str],
    concepts_to_clean: list[str],
) -> torch.Tensor:
    if cavs.ndim != 2 or activations.ndim != 2 or labels.ndim != 2:
        raise ValueError("CAVs, activations, and labels must all be two-dimensional.")
    if activations.shape[0] != labels.shape[0]:
        raise ValueError(
            "Activations and labels must contain the same number of samples, got "
            f"{activations.shape[0]} and {labels.shape[0]}."
        )
    if cavs.shape[1] != activations.shape[1]:
        raise ValueError(
            "CAVs and activations must contain the same number of features, got "
            f"{cavs.shape[1]} and {activations.shape[1]}."
        )
    if cavs.shape[0] < len(concept_names) or labels.shape[1] < len(concept_names):
        raise ValueError(
            "CAV and label concept dimensions must cover all concept names."
        )

    cleaned_cavs = cavs.clone()
    activations = activations.to(cavs)
    for concept_name in concepts_to_clean:
        if concept_name not in concept_names:
            raise ValueError(f"Unknown concept '{concept_name}' requested for cleaning.")
        concept_id = concept_names.index(concept_name)
        non_concept_mask = labels[:, concept_id] == 0
        if not non_concept_mask.any():
            raise ValueError(
                f"No non-concept samples are available for concept '{concept_name}'."
            )
        non_concept_mean = activations[non_concept_mask].mean(dim=0)
        cleaned_cavs[concept_id] = _clean_cav(
            cleaned_cavs[concept_id], non_concept_mean
        )
    return cleaned_cavs


def _build_cav_sets(
    cavs_baseline: torch.Tensor,
    cavs_orthogonal: torch.Tensor,
    concept_names: list[str],
    concepts_to_clean: list[str],
    cleaning_enabled: bool,
    activations: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    cav_sets = {
        "Baseline": cavs_baseline.cpu(),
        "Orthogonal": cavs_orthogonal.cpu(),
    }
    if not cleaning_enabled:
        return cav_sets
    if activations is None or labels is None:
        raise ValueError(
            "Concept cleaning requires preprocessing activations and labels."
        )

    cav_sets["Baseline Cleaned"] = _clean_cavs(
        cavs_baseline.cpu(),
        activations,
        labels,
        concept_names,
        concepts_to_clean,
    )
    cav_sets["Orthogonal Cleaned"] = _clean_cavs(
        cavs_orthogonal.cpu(),
        activations,
        labels,
        concept_names,
        concepts_to_clean,
    )
    return cav_sets


def _add_metric_errorbars(
    ax: plt.Axes, sem_lookup: dict[str, float], cav_order: list[str]
) -> None:
    for cav_name, container in zip(cav_order, ax.containers[: len(cav_order)]):
        if len(container) == 0:
            continue
        sem = sem_lookup.get(cav_name)
        if sem is None or np.isnan(sem):
            continue
        patch = container[0]
        x = patch.get_x() + patch.get_width() / 2
        y = patch.get_height()
        ax.errorbar(
            x,
            y,
            yerr=sem,
            fmt="none",
            ecolor="black",
            elinewidth=1,
            capsize=4,
            capthick=1,
        )


def _select_heatmap_samples(dataset, cfg: DictConfig) -> np.ndarray:
    artifact_keys = cfg.heatmaps.artifacts
    selected_ids = None
    for key in artifact_keys:
        ids = dataset.sample_ids_by_artifact.get(key)
        if ids is None:
            continue
        ids_np = np.array(ids)
        selected_ids = (
            ids_np if selected_ids is None else np.intersect1d(selected_ids, ids_np)
        )
    if selected_ids is None:
        selected_ids = np.arange(len(dataset))
    idxs_test = getattr(dataset, "idxs_test", None)
    if idxs_test is not None:
        selected_ids = np.intersect1d(selected_ids, np.array(idxs_test))
    return selected_ids


def _metric_sem(values: np.ndarray) -> float:
    if len(values) == 0:
        return float("nan")
    return float(values.std() / np.sqrt(len(values)))


def _store_metric_stats(
    results_quant: dict[str, float],
    metric_name: str,
    concept_name: str,
    cav_name: str,
    values: np.ndarray,
) -> None:
    results_quant[f"{metric_name}_{concept_name}_{cav_name}"] = float(values.mean())
    results_quant[f"{metric_name}_{concept_name}_{cav_name}_sem"] = _metric_sem(values)


def _dilate_binary_masks(masks: torch.Tensor, padding: int) -> torch.Tensor:
    if padding <= 0:
        return masks
    if masks.ndim != 3:
        raise ValueError(
            f"Expected masks with shape (N, H, W), got {tuple(masks.shape)}"
        )
    kernel_size = 2 * padding + 1
    dilated = F.max_pool2d(
        masks.float().unsqueeze(1),
        kernel_size=kernel_size,
        stride=1,
        padding=padding,
    ).squeeze(1)
    return dilated.to(dtype=masks.dtype)


def _build_metric_plot_frames(
    results_quant: dict[str, float],
    metric_name: str,
    concept_name: str,
    metric_label: str,
    cav_order: list[str] = CAV_PLOT_ORDER,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_plot = []
    data_plot_std = []
    for cav_name in cav_order:
        mean = results_quant.get(f"{metric_name}_{concept_name}_{cav_name}", 0.0)
        sem = results_quant.get(f"{metric_name}_{concept_name}_{cav_name}_sem", 0.0)
        data_plot.append((cav_name, mean))
        data_plot_std.append((cav_name, mean, sem))
    return (
        pd.DataFrame(data_plot, columns=["CAV", metric_label]),
        pd.DataFrame(data_plot_std, columns=["CAV", metric_label, "SEM"]),
    )


def _save_metric_plot(
    data_plot: pd.DataFrame,
    metric_label: str,
    savename: Path,
    ymin: float,
    ymax: float,
    ticks: list[float],
    cav_order: list[str] = CAV_PLOT_ORDER,
) -> None:
    sns.set_style("whitegrid")
    plt.rcParams.update({"font.size": 9, "legend.fontsize": 9, "axes.titlesize": 11})
    fig, ax = plt.subplots(figsize=(max(2.5, 1.25 * len(cav_order)), 3))
    sns.barplot(
        x="CAV",
        y=metric_label,
        hue="CAV",
        data=data_plot,
        order=cav_order,
        hue_order=cav_order,
        ax=ax,
    )
    ax.set_ylabel(metric_label)
    ax.set_ylim(ymin, ymax)
    ax.set_yticks(ticks)
    ax.xaxis.grid(False)
    ax.yaxis.grid(True)
    [
        fig.savefig(f"{savename}.{ending}", bbox_inches="tight", dpi=500)
        for ending in ["png", "pdf"]
    ]
    plt.close(fig)


def _save_metric_plot_std(
    data_plot: pd.DataFrame,
    metric_label: str,
    savename: Path,
    ymin: float,
    ymax: float,
    ticks: list[float],
    cav_order: list[str] = CAV_PLOT_ORDER,
) -> None:
    sns.set_style("whitegrid")
    plt.rcParams.update({"font.size": 9, "legend.fontsize": 9, "axes.titlesize": 11})
    fig, ax = plt.subplots(figsize=(max(2.5, 1.25 * len(cav_order)), 3))
    ordered_df = data_plot.set_index("CAV").reindex(cav_order).reset_index()
    sns.barplot(
        x="CAV",
        y=metric_label,
        hue="CAV",
        data=ordered_df,
        order=cav_order,
        hue_order=cav_order,
        errorbar=None,
        ax=ax,
    )
    _add_metric_errorbars(
        ax,
        {row["CAV"]: float(row["SEM"]) for _, row in ordered_df.iterrows()},
        cav_order,
    )
    ax.set_ylabel(metric_label)
    ax.set_ylim(ymin, ymax)
    ax.set_yticks(ticks)
    ax.xaxis.grid(False)
    ax.yaxis.grid(True)
    [
        fig.savefig(f"{savename}.{ending}", bbox_inches="tight", dpi=500)
        for ending in ["png", "pdf"]
    ]
    plt.close(fig)


def evaluate_concept_heatmaps(
    cfg: DictConfig,
    cav_model: nn.Module,
    base_model: nn.Module,
    num_imgs: int = 16,
    *,
    activations: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
) -> None:
    device = torch.device(cfg.train.device)
    dataset = get_dataset(cfg.dataset.name + "_hm")(**cfg.dataset)
    classification_model = load_base_model(cfg, dataset.num_classes, device)
    cavs_baseline, _ = base_model.get_params()  # type: ignore
    cavs_orthogonal, _ = cav_model.get_params()  # type: ignore
    concept_names = dataset.get_concept_names()
    concepts_to_plot = []
    for cname in list(cfg.heatmaps.concepts):
        if cname not in concept_names:
            log.warning(
                f"Concept {cname} not found in dataset concepts. Available concepts: {concept_names}"
            )
        else:
            concepts_to_plot.append(cname)
    if len(concepts_to_plot) == 0:
        log.warning("No requested heatmap concepts are available. Skipping.")
        return

    sample_ids = _select_heatmap_samples(dataset, cfg)
    if len(sample_ids) == 0:
        log.warning(
            "No samples matched the requested artifact configuration. Skipping heatmap evaluation."
        )
        return
    ds_subset = dataset.get_subset_by_idxs(sample_ids.tolist())

    cleaning_cfg = cfg.heatmaps.get("concept_cleaning", None)
    cleaning_enabled = bool(
        cleaning_cfg is not None and cleaning_cfg.get("enabled", False)
    )
    cav_sets = _build_cav_sets(
        cavs_baseline,
        cavs_orthogonal,
        concept_names,
        concepts_to_plot,
        cleaning_enabled,
        activations,
        labels,
    )
    cav_order = list(cav_sets.keys())
    if cleaning_enabled:
        log.info("Using post-hoc concept-cleaned CAVs for LRP localization.")

    if requires_lxt_localization(cfg.model.name):
        composite = None
        attribution = None
    else:
        canonizers = get_canonizer(cfg.model.name)
        composite = EpsilonPlusFlat(canonizers=canonizers)
        attribution = CondAttribution(classification_model)

    save_dir = get_save_dir(cfg)
    results_dir = save_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    cav_localizations: Dict[str, Dict[str, torch.Tensor]] = {}
    imgs = None
    gts = None
    for name, cavs in cav_sets.items():
        cav_subset = {
            cname: cavs[concept_names.index(cname), :] for cname in concepts_to_plot
        }
        imgs, localizations, gts = compute_concept_relevances(
            classification_model,
            attribution,
            ds_subset,
            cav_subset,
            composite,
            cfg,
            device,
            batch_size=cfg.train.batch_size,
        )
        cav_localizations[name] = localizations  # type: ignore

    if imgs is None or gts is None:
        log.warning(
            "Failed to compute heatmaps due to missing ground-truth annotations."
        )
        return

    savepath = results_dir / f"concept_heatmaps"
    create_plot(
        dataset,
        imgs[: min(len(imgs), num_imgs)],
        cav_localizations,
        gts,
        concepts_to_plot,
        savepath,
    )

    results_quant = {}
    metric_masks = {
        cname: gts[cname]
        for cname in concepts_to_plot
        if cname in gts and gts[cname] is not None
    }
    if "box" in metric_masks:
        metric_masks[BOX_DILATED_CONCEPT_NAME] = _dilate_binary_masks(
            metric_masks["box"], BOX_MASK_DILATION_PIXELS
        )
    for cav_name, locs in cav_localizations.items():
        for cname, gt_mask in metric_masks.items():
            loc_name = "box" if cname == BOX_DILATED_CONCEPT_NAME else cname
            if loc_name not in locs:
                continue
            loc = locs[loc_name]
            concept_rel = compute_concept_relevance(loc, gt_mask)
            concept_rel_np = concept_rel.numpy()
            loc_binary = binarize_heatmaps(loc, thresholding="otsu").bool()
            gt_binary = gt_mask.bool()
            intersection = torch.logical_and(loc_binary, gt_binary).sum((1, 2)).float()
            union = torch.logical_or(loc_binary, gt_binary).sum((1, 2)).float()
            gt_area = gt_binary.sum((1, 2)).float()
            ious = (intersection / (union + 1e-10)).numpy()
            inter_over_true_mask = (intersection / (gt_area + 1e-10)).numpy()

            _store_metric_stats(
                results_quant, "concept_rel", cname, cav_name, concept_rel_np
            )
            _store_metric_stats(results_quant, "iou", cname, cav_name, ious)
            _store_metric_stats(
                results_quant,
                "intersection_over_true_mask",
                cname,
                cav_name,
                inter_over_true_mask,
            )

    for cname in metric_masks:
        metric_prefix = "" if cname == "timestamp" else f"{cname}_"
        data_plot, data_plot_std = _build_metric_plot_frames(
            results_quant,
            "concept_rel",
            cname,
            "Concept Relevance",
            cav_order,
        )
        vmax = 0.5 if data_plot["Concept Relevance"].max() > 0.44 else 0.45
        savepath_quant = results_dir / f"{metric_prefix}concept_relevance"
        plot_concept_relevance(data_plot, vmax, savepath_quant, cav_order)
        plot_concept_relevance_std(
            data_plot_std,
            vmax,
            results_dir / f"{metric_prefix}concept_relevance_std",
            cav_order,
        )
        for metric_name, metric_label in [
            ("iou", "IoU"),
            ("intersection_over_true_mask", "Intersection over True Mask"),
        ]:
            data_plot, data_plot_std = _build_metric_plot_frames(
                results_quant, metric_name, cname, metric_label, cav_order
            )
            plot_overlap_metric(
                data_plot,
                metric_label,
                results_dir / f"{metric_prefix}{metric_name}",
                cav_order,
            )
            plot_overlap_metric_std(
                data_plot_std,
                metric_label,
                results_dir / f"{metric_prefix}{metric_name}_std",
                cav_order,
            )

    with open(results_dir / f"concept_relevance.pkl", "wb") as f:
        pickle.dump(results_quant, f)


def plot_concept_relevance(
    data_plot: pd.DataFrame,
    vmax: float,
    savename: Path,
    cav_order: list[str] = CAV_PLOT_ORDER,
) -> None:
    ticks = [0.25, 0.3, 0.35, 0.4, 0.45]
    if vmax == 0.5:
        ticks.append(0.5)
    _save_metric_plot(
        data_plot, "Concept Relevance", savename, 0.25, vmax, ticks, cav_order
    )


def plot_concept_relevance_std(
    data_plot: pd.DataFrame,
    vmax: float,
    savename: Path,
    cav_order: list[str] = CAV_PLOT_ORDER,
) -> None:
    ymax = float((data_plot["Concept Relevance"] + data_plot["SEM"]).max())
    vmax_plot = max(vmax, ymax + 0.01)
    ymin = 0.25 if float(data_plot["Concept Relevance"].min()) >= 0.25 else 0.0
    if ymin == 0.25:
        ticks = [0.25, 0.3, 0.35, 0.4, 0.45]
        if vmax_plot >= 0.5:
            ticks.append(0.5)
    else:
        vmax_plot = max(0.05, float(np.ceil(vmax_plot / 0.05) * 0.05))
        ticks = np.arange(ymin, vmax_plot + 1e-9, 0.05).tolist()
    _save_metric_plot_std(
        data_plot,
        "Concept Relevance",
        savename,
        ymin,
        vmax_plot,
        ticks,
        cav_order,
    )


def plot_overlap_metric(
    data_plot: pd.DataFrame,
    metric_label: str,
    savename: Path,
    cav_order: list[str] = CAV_PLOT_ORDER,
) -> None:
    vmax = min(
        1.0, max(0.1, float(np.ceil(float(data_plot[metric_label].max()) / 0.1) * 0.1))
    )
    ticks = np.arange(0.0, vmax + 1e-9, 0.1).tolist()
    _save_metric_plot(
        data_plot, metric_label, savename, 0.0, vmax, ticks, cav_order
    )


def plot_overlap_metric_std(
    data_plot: pd.DataFrame,
    metric_label: str,
    savename: Path,
    cav_order: list[str] = CAV_PLOT_ORDER,
) -> None:
    ymax = float((data_plot[metric_label] + data_plot["SEM"]).max())
    vmax = min(1.0, max(0.1, float(np.ceil(ymax / 0.1) * 0.1)))
    ticks = np.arange(0.0, vmax + 1e-9, 0.1).tolist()
    _save_metric_plot_std(
        data_plot, metric_label, savename, 0.0, vmax, ticks, cav_order
    )


def compute_concept_relevances(
    classification_model: nn.Module,
    attribution,
    ds,
    cavs: Dict[str, torch.Tensor],
    composite,
    cfg: DictConfig,
    device: torch.device,
    batch_size: int = 8,
):
    localizations = {c: None for c in cavs.keys()}
    artifact_names = list(cfg.heatmaps.artifacts)
    gts = {c: None for c in artifact_names}
    layer_name = cfg.cav.layer
    hm_config = {"layer_name": layer_name}
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    imgs = None
    for batch in tqdm.tqdm(dl):
        x = batch[0]
        artifact_masks = batch[2]
        if not isinstance(artifact_masks, dict):
            if len(batch) == 4:
                artifact_masks = {"timestamp": batch[2], "box": batch[3]}
            else:
                raise ValueError(
                    "Expected heatmap dataset to return artifact masks as a dict."
                )
        for cname, cav in cavs.items():
            _, loc_cav = get_localizations(
                x.clone(),
                cav,
                attribution,
                composite,
                hm_config,
                device,
                model_name=cfg.model.name,
                model=classification_model,
            )
            localizations[cname] = (
                loc_cav
                if localizations[cname] is None
                else torch.cat([localizations[cname], loc_cav])
            )
        for cname in artifact_names:
            if cname not in artifact_masks:
                continue
            mask = artifact_masks[cname]
            gts[cname] = mask if gts[cname] is None else torch.cat([gts[cname], mask])
        imgs = x.detach().cpu() if imgs is None else torch.cat([imgs, x.detach().cpu()])
    return imgs, localizations, gts


def create_plot(
    ds, imgs, cav_localizations, gts, concepts_to_plot, savepath: Path
) -> None:
    num_cavs = len(cav_localizations)
    nrows = len(imgs)
    plotted_concepts = [
        cname
        for cname in concepts_to_plot
        if any(cname in localizations for localizations in cav_localizations.values())
    ]
    ncols = 1 + sum(
        num_cavs + (1 if cname in gts and gts[cname] is not None else 0)
        for cname in plotted_concepts
    )
    size = 1.7
    level = 2.0
    fig, axs = plt.subplots(
        nrows, ncols, figsize=(ncols * size, nrows * size), squeeze=False
    )

    for i in range(nrows):
        ax = axs[i][0]
        ax.imshow(ds.reverse_normalization(imgs[i]).permute((1, 2, 0)).int().numpy())
        axs[0][0].set_title("Input")

        c = 1
        for cname in plotted_concepts:
            all_maxs = [
                all_concept_hms[cname][i].max().detach().float()
                for _, all_concept_hms in cav_localizations.items()
                if cname in all_concept_hms
            ]
            normalization_constant = torch.stack(all_maxs).max().clamp_min(1e-12)
            for cav_name, localizations in cav_localizations.items():
                ax = axs[i][c]
                img_hm = imgify(
                    localizations[cname][i] / normalization_constant,
                    cmap="bwr",
                    vmin=-1,
                    vmax=1,
                    level=level,
                )
                ax.imshow(img_hm)
                axs[0][c].set_title(f"{cname}\n{cav_name}")
                c += 1
            if cname in gts and gts[cname] is not None:
                ax = axs[i][c]
                ax.imshow(gts[cname][i].numpy())
                axs[0][c].set_title(f"{cname}\nGround Truth")
                c += 1

    for _axs in axs:
        for ax in _axs:
            ax.set_xticks([])
            ax.set_yticks([])

    log.info("Storing heatmaps at %s", savepath)
    [
        fig.savefig(f"{savepath}.{ending}", bbox_inches="tight")
        for ending in ["png", "pdf"]
    ]
    plt.close(fig)
