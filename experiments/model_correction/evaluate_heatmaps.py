import logging
import pickle
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import tqdm
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader

from crp.attribution import CondAttribution
from crp.image import imgify
from models import get_canonizer
from sklearn.metrics import jaccard_score
from zennit.composites import EpsilonPlusFlat

from experiments.model_correction.utils import load_base_model
from experiments.utils.activations import extract_latents
from experiments.utils.utils import get_save_dir
from utils.cav import compute_cavs
from utils.localization import get_localizations, binarize_heatmaps
from datasets import get_dataset

log = logging.getLogger(__name__)


def _select_heatmap_samples(dataset, cfg: DictConfig) -> np.ndarray:
    artifact_keys = cfg.heatmaps.artifacts
    selected_ids = None
    for key in artifact_keys:
        ids = dataset.sample_ids_by_artifact.get(key)
        if ids is None:
            continue
        ids_np = np.array(ids)
        selected_ids = ids_np if selected_ids is None else np.intersect1d(selected_ids, ids_np)
    if selected_ids is None:
        selected_ids = np.arange(len(dataset))
    idxs_test = getattr(dataset, "idxs_test", None)
    if idxs_test is not None:
        selected_ids = np.intersect1d(selected_ids, np.array(idxs_test))
    return selected_ids


def evaluate_concept_heatmaps(
    cfg: DictConfig, cav_model: nn.Module, base_model:nn.Module, num_imgs: int = 16) -> None:
    device = torch.device(cfg.train.device)
    dataset = get_dataset(cfg.dataset.name + "_hm")(**cfg.dataset)
    classification_model = load_base_model(cfg, dataset.num_classes, device)
    cavs_baseline, _ = base_model.get_params()   # type: ignore
    cavs_orthogonal, _ = cav_model.get_params() # type: ignore
    cav_sets = {
        "Baseline": cavs_baseline.cpu(),
        "Orthogonal": cavs_orthogonal.cpu(),
    }
    concept_names = dataset.get_concept_names()
    concepts_to_plot = cfg.heatmaps.concepts
    for cname in concepts_to_plot:
        if cname not in concept_names:
            log.warning(f"Concept {cname} not found in dataset concepts. Available concepts: {concept_names}")

    sample_ids = _select_heatmap_samples(dataset, cfg)
    if len(sample_ids) == 0:
        log.warning("No samples matched the requested artifact configuration. Skipping heatmap evaluation.")
        return
    ds_subset = dataset.get_subset_by_idxs(sample_ids.tolist())
    attribution = CondAttribution(classification_model)
    canonizers = get_canonizer(cfg.model.name)
    composite = EpsilonPlusFlat(canonizers)

    save_dir = get_save_dir(cfg)
    results_dir = save_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    cav_localizations: Dict[str, Dict[str, torch.Tensor]] = {}
    imgs = None
    gts = None
    for name, cavs in cav_sets.items():
        cav_subset = {cname: cavs[concept_names.index(cname), :] for cname in concepts_to_plot}
        imgs, localizations, gts = compute_concept_relevances(
            attribution,
            ds_subset,
            cav_subset,
            composite,
            cfg,
            device,
            batch_size=cfg.train.batch_size,
        )
        cav_localizations[name] = localizations        # type: ignore

    if imgs is None or gts is None:
        log.warning("Failed to compute heatmaps due to missing ground-truth annotations.")
        return

    savepath = results_dir / f"concept_heatmaps"
    create_plot(dataset, imgs[: min(len(imgs), num_imgs)], cav_localizations, gts["timestamp"], gts["box"], savepath)

    results_quant = {}
    for cav_name, locs in cav_localizations.items():
        for cname in ["timestamp", "box"]:
            if cname not in locs:
                continue
            loc = locs[cname]
            concept_rel = ((loc * gts[cname]).sum((1, 2)) / (loc.sum((1, 2)) + 1e-10))
            loc_binary = binarize_heatmaps(loc, thresholding="otsu")
            jaccards = np.array([
                jaccard_score(loc_binary[i].reshape(-1).numpy(), gts[cname][i].reshape(-1).numpy())
                for i in range(len(loc_binary))
            ])
            results_quant[f"iou_{cname}_{cav_name}"] = jaccards.mean()
            results_quant[f"concept_rel_{cname}_{cav_name}"] = concept_rel.mean().item()

    data_plot = pd.DataFrame(data=[
        ("Baseline", results_quant.get(f"concept_rel_timestamp_Baseline", 0.0)),
        ("Orthogonal", results_quant.get(f"concept_rel_timestamp_Orthogonal", 0.0)),
    ], columns=["CAV", "Concept Relevance"])
    vmax = 0.5 if data_plot["Concept Relevance"].max() > 0.44 else 0.45
    savepath_quant = results_dir / f"concept_relevance"
    plot_concept_relevance(data_plot, vmax, savepath_quant)

    with open(results_dir / f"concept_relevance.pkl", "wb") as f:
        pickle.dump(results_quant, f)


def plot_concept_relevance(data_plot: pd.DataFrame, vmax: float, savename: Path) -> None:
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 9, 'legend.fontsize': 9, 'axes.titlesize': 11})
    fig = plt.figure(figsize=(2.5, 3))
    sns.barplot(x='CAV', y='Concept Relevance', hue="CAV", data=data_plot)
    ticks = [0.25, 0.3, 0.35, 0.4, 0.45]
    if vmax == .5:
        ticks.append(0.5)
    plt.ylim(0.25, vmax)
    plt.yticks(ticks)
    [fig.savefig(f"{savename}.{ending}", bbox_inches="tight", dpi=500) for ending in ["png", "pdf"]]


def compute_concept_relevances(
    attribution: CondAttribution,
    ds,
    cavs: Dict[str, torch.Tensor],
    composite,
    cfg: DictConfig,
    device: torch.device,
    batch_size: int = 8,
):
    localizations = {c: None for c in cavs.keys()}
    gts = {"timestamp": None, "box": None}
    layer_name = cfg.cav.layer
    hm_config = {"layer_name": layer_name}
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    imgs = None
    for x, _, loc_timestamp, loc_box in tqdm.tqdm(dl):
        for cname, cav in cavs.items():
            attr, loc_cav = get_localizations(x.clone(), cav, attribution, composite, hm_config, device)
            loc_cav = attr.heatmap.detach().cpu().clamp(min=0)
            localizations[cname] = loc_cav if localizations[cname] is None else torch.cat(
                [localizations[cname], loc_cav]
            )
        gts["timestamp"] = loc_timestamp if gts["timestamp"] is None else torch.cat([gts["timestamp"], loc_timestamp])
        gts["box"] = loc_box if gts["box"] is None else torch.cat([gts["box"], loc_box])
        imgs = x.detach().cpu() if imgs is None else torch.cat([imgs, x.detach().cpu()])
    return imgs, localizations, gts


def create_plot(ds, imgs, cav_localizations, gt_timestamp, gt_box, savepath: Path) -> None:
    num_cavs = len(cav_localizations)
    nrows = len(imgs)
    ncols = 3 + 3 * num_cavs
    size = 1.7
    level = 2.0
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * size, nrows * size))

    for i in range(nrows):
        ax = axs[i][0]
        ax.imshow(ds.reverse_normalization(imgs[i]).permute((1, 2, 0)).int().numpy())
        axs[0][0].set_title("Input")

        for cav_idx, (cav_name, localizations) in enumerate(cav_localizations.items()):
            cname = "timestamp"
            all_maxs = [all_concept_hms[cname][i].max() for _, all_concept_hms in cav_localizations.items()]
            normalization_constant = torch.max(torch.tensor(all_maxs))
            c = 1 + cav_idx
            ax = axs[i][c]
            img_hm = imgify(localizations[cname][i] / normalization_constant, cmap="bwr", vmin=-1, vmax=1, level=level)
            ax.imshow(img_hm)
            axs[0][c].set_title(f"{cname}\n{cav_name}")

            cname = "box"
            all_maxs = [all_concept_hms[cname][i].max() for _, all_concept_hms in cav_localizations.items()]
            normalization_constant = torch.max(torch.tensor(all_maxs))
            c = 1 + num_cavs + 1 + cav_idx
            ax = axs[i][c]
            img_hm = imgify(localizations[cname][i] / normalization_constant, cmap="bwr", vmin=-1, vmax=1, level=level)
            ax.imshow(img_hm)
            axs[0][c].set_title(f"{cname}\n{cav_name}")

            cname = "Blond_Hair"
            if cname in localizations:
                all_maxs = [all_concept_hms[cname][i].max() for _, all_concept_hms in cav_localizations.items()]
                normalization_constant = torch.max(torch.tensor(all_maxs))
                c = 1 + 2 * (num_cavs + 1) + cav_idx
                ax = axs[i][c]
                img_hm = imgify(localizations[cname][i] / normalization_constant, cmap="bwr", vmin=-1, vmax=1, level=level)
                ax.imshow(img_hm)
                axs[0][c].set_title(f"{cname}\n{cav_name}")

        c = 1 + num_cavs
        ax = axs[i][c]
        ax.imshow(gt_timestamp[i].numpy())
        axs[0][c].set_title("Ground Truth")

        c = 1 + 2 * num_cavs + 1
        ax = axs[i][c]
        ax.imshow(gt_box[i].numpy())
        axs[0][c].set_title("Ground Truth")

    for _axs in axs:
        for ax in _axs:
            ax.set_xticks([])
            ax.set_yticks([])

    log.info("Storing heatmaps at %s", savepath)
    [fig.savefig(f"{savepath}.{ending}", bbox_inches="tight") for ending in ["png", "pdf"]]
