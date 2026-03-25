#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from crp.attribution import CondAttribution
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from zennit.composites import EpsilonPlusFlat

from datasets import get_dataset
from experiments.model_correction.dir_model import (
    load_base_cav_model,
    load_dir_model,
)
from experiments.model_correction.evaluate_heatmaps import (
    _is_vit_model,
    _select_heatmap_samples,
    compute_concept_relevances,
)
from experiments.model_correction.utils import load_base_model
from experiments.utils.activations import _get_features
from models import get_canonizer


MODEL_ORDER = ["Vanilla", "Baseline CAV", "Orthogonal CAV"]
CATEGORY_ORDER = ["attacked", "clean"]
MODEL_KEY_BY_DISPLAY = {
    "Vanilla": "vanilla",
    "Baseline CAV": "baseline",
    "Orthogonal CAV": "orthogonal",
}


def _add_grouped_bar_errorbars(
    ax: plt.Axes,
    sem_lookup: dict[tuple[str, str], float],
    category_order: list[str],
    model_order: list[str],
) -> None:
    for model_name, container in zip(model_order, ax.containers[: len(model_order)]):
        for category_name, patch in zip(category_order, container):
            sem = sem_lookup.get((category_name, model_name))
            if sem is None or np.isnan(sem):
                continue
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


def _add_concept_relevance_errorbars(
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


@dataclass(frozen=True)
class RunSpec:
    model_name: str
    layer_name: str
    cav_name: str
    alpha: float
    learning_rate: float
    optimal_init: bool
    beta: float | None
    n_targets: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create SEM-aware model-correction figures for every results directory."
        )
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/model_correction"),
        help="Root directory that contains model correction runs.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device used when concept relevance needs to be recomputed.",
    )
    parser.add_argument(
        "--dataset-name",
        default="celeba_attacked",
        help="Dataset config name used for these runs.",
    )
    return parser.parse_args()


def find_result_dirs(results_root: Path) -> list[Path]:
    result_dirs = sorted(
        path.parent
        for path in results_root.rglob("selected_metrics.pkl")
        if (path.parent / "concept_relevance.pkl").exists()
        and (path.parent / "metrics_per_model.pkl").exists()
        and (path.parent / "confusion_per_model.pkl").exists()
    )
    return result_dirs


def parse_run_spec(result_dir: Path) -> RunSpec:
    run_dir = result_dir.parent
    run_name = run_dir.name.split("--", 1)[0]
    match = re.fullmatch(
        (
            r"alpha(?P<alpha>[^_]+)"
            r"(?:_beta(?P<beta>[^_]+)_n_targets(?P<n_targets>\d+))?"
            r"_lr(?P<lr>[^_]+)"
            r"(?P<opt>_opt)?"
        ),
        run_name,
    )
    if match is None:
        raise ValueError(f"Could not parse run name: {run_dir.name}")

    return RunSpec(
        model_name=result_dir.parents[3].name,
        layer_name=result_dir.parents[2].name,
        cav_name=result_dir.parents[1].name,
        alpha=float(match.group("alpha")),
        learning_rate=float(match.group("lr")),
        optimal_init=match.group("opt") is not None,
        beta=float(match.group("beta")) if match.group("beta") is not None else None,
        n_targets=int(match.group("n_targets") or 0),
    )


def build_cfg(repo_root: Path, spec: RunSpec, dataset_name: str, device: str) -> DictConfig:
    base_cfg = OmegaConf.load(repo_root / "configs/model_correction.yaml")
    hardware_cfg = OmegaConf.load(repo_root / "configs/hardware/workstation.yaml")
    dataset_cfg = OmegaConf.load(repo_root / f"configs/dataset/{dataset_name}.yaml")
    model_cfg = OmegaConf.load(repo_root / f"configs/model/{spec.model_name}.yaml")
    cav_cfg = OmegaConf.load(repo_root / f"configs/cav_model/{spec.cav_name}.yaml")

    train_cfg = OmegaConf.merge(base_cfg.train, hardware_cfg)
    train_cfg.learning_rate = spec.learning_rate
    train_cfg.device = device

    merged_cav = OmegaConf.merge(base_cfg.cav, cav_cfg)
    merged_cav.layer = spec.layer_name
    merged_cav.alpha = spec.alpha
    merged_cav.beta = spec.beta
    merged_cav.n_targets = spec.n_targets
    merged_cav.optimal_init = spec.optimal_init

    dataset_cfg.val_split = train_cfg.val_ratio
    dataset_cfg.test_split = train_cfg.test_ratio
    dataset_cfg.seed = train_cfg.random_seed
    dataset_cfg.name = dataset_name

    cfg = OmegaConf.create(
        {
            "experiment": {
                "name": "model_correction",
                "out": "",
                "random_seed": base_cfg.experiment.random_seed,
            },
            "train": train_cfg,
            "dataset": dataset_cfg,
            "model": model_cfg,
            "cav": merged_cav,
            "correction": {
                "method": base_cfg.correction.method,
                "dir_precomputed_data": (
                    f"variables/{dataset_name}/{spec.model_name}"
                ),
                "layer_name": spec.layer_name,
                "dataset_name": dataset_name,
                "model_name": spec.model_name,
                "direction_mode": spec.cav_name,
            },
            "heatmaps": base_cfg.heatmaps,
        }
    )
    return cfg


def build_metric_df(result_dir: Path, metric_type: str) -> pd.DataFrame:
    selected_metrics = pd.read_pickle(result_dir / "selected_metrics.pkl")
    metrics_per_model: dict[str, dict[str, float]] = pd.read_pickle(
        result_dir / "metrics_per_model.pkl"
    )
    confusion_per_model: dict[str, dict[str, Any]] = pd.read_pickle(
        result_dir / "confusion_per_model.pkl"
    )

    df = (
        selected_metrics.loc[selected_metrics["Metric Type"] == metric_type]
        .copy()
        .reset_index(drop=True)
    )
    if metric_type == "test_accuracy":
        df["SEM"] = df.apply(
            lambda row: metrics_per_model[MODEL_KEY_BY_DISPLAY[row["Model"]]][
                f"test_accuracy_standard_err_{row['Category']}"
            ],
            axis=1,
        )
        return df

    if metric_type == "test_fnr":
        sems = []
        for _, row in df.iterrows():
            model_key = MODEL_KEY_BY_DISPLAY[row["Model"]]
            confusion = confusion_per_model[model_key]["test"][row["Category"]]
            metric_match = re.search(r"test_fnr_(\d+)_", row["Metric"])
            if metric_match is None:
                raise ValueError(f"Could not infer class id from metric {row['Metric']}")
            class_id = int(metric_match.group(1))
            n_positives = float(confusion[class_id].sum())
            value = float(row["Value"])
            sem = math.sqrt(value * (1.0 - value) / n_positives) if n_positives else 0.0
            sems.append(sem)
        df["SEM"] = sems
        return df

    raise ValueError(f"Unsupported metric type: {metric_type}")


def plot_metric_with_sem(metric_df: pd.DataFrame, metric_type: str, save_dir: Path) -> None:
    sns.set_style("whitegrid")
    plt.rcParams.update({"font.size": 9, "legend.fontsize": 9, "axes.titlesize": 11})

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(
        x="Category",
        y="Value",
        hue="Model",
        data=metric_df,
        order=CATEGORY_ORDER,
        hue_order=MODEL_ORDER,
        errorbar=None,
        ax=ax,
    )
    sem_lookup = {
        (row["Category"], row["Model"]): float(row["SEM"])
        for _, row in metric_df.iterrows()
    }
    _add_grouped_bar_errorbars(ax, sem_lookup, CATEGORY_ORDER, MODEL_ORDER)

    label_map = {
        "test_accuracy": "Accuracy",
        "test_fnr": "False Negative Rate",
    }
    ax.set_xlabel("")
    ax.set_ylabel(label_map[metric_type])

    if metric_type == "test_accuracy":
        ax.set_ylim(0.5, 0.95)
        ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9])

    ax.xaxis.grid(False)
    ax.yaxis.grid(True)
    ax.legend(title="", loc="upper left", bbox_to_anchor=(-0.22, -0.15), ncols=3)
    fig.savefig(
        save_dir / f"metric_comparison_{metric_type}_std.png",
        bbox_inches="tight",
        dpi=500,
    )
    fig.savefig(
        save_dir / f"metric_comparison_{metric_type}_std.pdf",
        bbox_inches="tight",
        dpi=500,
    )
    plt.close(fig)


def load_or_extract_latents(
    repo_root: Path,
    cfg: DictConfig,
    model: torch.nn.Module,
    dataset: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    cache_path = (
        repo_root
        / "variables"
        / str(cfg.dataset.name)
        / str(cfg.model.name)
        / f"{cfg.cav.layer}.pth"
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        variables = torch.load(cache_path, weights_only=True)
        return variables["encs"], variables["labels"]

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=False,
    )

    if _is_vit_model(cfg.model.name):
        import zennit.rules as z_rules
        from zennit.composites import LayerMapComposite

        composite = LayerMapComposite(
            [
                (torch.nn.Conv2d, z_rules.Gamma(100)),
                (torch.nn.Linear, z_rules.Gamma(0.1)),
            ]
        )
    else:
        canonizers = get_canonizer(cfg.model.name)
        composite = EpsilonPlusFlat(canonizers=canonizers)

    attribution = CondAttribution(model)
    latents = []
    for x_batch, _ in tqdm(dataloader):
        batch_latents = _get_features(
            x_batch,
            cfg.cav.layer,
            attribution,
            composite,
            cfg.cav.cav_mode,
            device=torch.device(cfg.train.device),
        )
        latents.append(batch_latents.detach().cpu())

    x_latent_all = torch.cat(latents)
    labels = dataset.get_labels().clamp(min=0)  # type: ignore[attr-defined]
    torch.save({"encs": x_latent_all, "labels": labels}, cache_path)
    return x_latent_all, labels


def compute_concept_relevance_df(result_dir: Path, cfg: DictConfig) -> pd.DataFrame:
    device = torch.device(cfg.train.device)
    heatmap_batch_size = min(2, int(cfg.train.batch_size))
    if device.type == "cuda":
        torch.cuda.empty_cache()
    stored_concept_relevance: dict[str, float] = pd.read_pickle(
        result_dir / "concept_relevance.pkl"
    )
    if all(
        key in stored_concept_relevance
        for key in [
            "concept_rel_timestamp_Baseline",
            "concept_rel_timestamp_Baseline_sem",
            "concept_rel_timestamp_Orthogonal",
            "concept_rel_timestamp_Orthogonal_sem",
        ]
    ):
        return pd.DataFrame(
            [
                {
                    "CAV": "Baseline",
                    "Concept Relevance": float(
                        stored_concept_relevance["concept_rel_timestamp_Baseline"]
                    ),
                    "SEM": float(
                        stored_concept_relevance["concept_rel_timestamp_Baseline_sem"]
                    ),
                },
                {
                    "CAV": "Orthogonal",
                    "Concept Relevance": float(
                        stored_concept_relevance["concept_rel_timestamp_Orthogonal"]
                    ),
                    "SEM": float(
                        stored_concept_relevance[
                            "concept_rel_timestamp_Orthogonal_sem"
                        ]
                    ),
                },
            ]
        )
    heatmap_dataset = get_dataset(cfg.dataset.name + "_hm")(**cfg.dataset)
    classification_model = load_base_model(cfg, heatmap_dataset.num_classes, device)
    concept_names = heatmap_dataset.get_concept_names()

    sample_ids = _select_heatmap_samples(heatmap_dataset, cfg)
    if len(sample_ids) == 0:
        raise RuntimeError(f"No heatmap samples found for {result_dir}.")
    ds_subset = heatmap_dataset.get_subset_by_idxs(sample_ids.tolist())

    if _is_vit_model(cfg.model.name):
        import zennit.rules as z_rules
        from zennit.composites import LayerMapComposite

        composite = LayerMapComposite(
            [
                (torch.nn.Conv2d, z_rules.Gamma(100)),
                (torch.nn.Linear, z_rules.Gamma(0.1)),
            ]
        )
    else:
        canonizers = get_canonizer(cfg.model.name)
        composite = EpsilonPlusFlat(canonizers=canonizers)

    attribution = CondAttribution(classification_model)

    dataset = instantiate(cfg.dataset)
    activations, labels = load_or_extract_latents(
        REPO_ROOT, cfg, classification_model, dataset
    )
    base_model = load_base_cav_model(cfg, activations, labels)
    dir_model = load_dir_model(cfg.cav._target_, result_dir.parent / "state_dict.pth")

    cav_sets = {
        "Baseline": base_model.get_params()[0].cpu(),
        "Orthogonal": dir_model.get_params()[0].cpu(),
    }
    timestamp_idx = concept_names.index("timestamp")

    rows: list[dict[str, float | str]] = []
    for cav_label, cavs in cav_sets.items():
        cav_subset = {"timestamp": cavs[timestamp_idx, :]}
        _, localizations, gts = compute_concept_relevances(
            attribution,
            ds_subset,
            cav_subset,
            composite,
            cfg,
            device,
            batch_size=heatmap_batch_size,
        )
        relevance_values = (
            (localizations["timestamp"] * gts["timestamp"]).sum((1, 2))
            / (localizations["timestamp"].sum((1, 2)) + 1e-10)
        ).numpy()
        mean_key = f"concept_rel_timestamp_{cav_label}"
        rows.append(
            {
                "CAV": cav_label,
                "Concept Relevance": float(
                    stored_concept_relevance.get(mean_key, relevance_values.mean())
                ),
                "SEM": float(relevance_values.std() / np.sqrt(len(relevance_values))),
            }
        )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def plot_concept_relevance_with_sem(concept_df: pd.DataFrame, save_dir: Path) -> None:
    sns.set_style("whitegrid")
    plt.rcParams.update({"font.size": 9, "legend.fontsize": 9, "axes.titlesize": 11})

    fig, ax = plt.subplots(figsize=(2.5, 3))
    cav_order = ["Baseline", "Orthogonal"]
    ordered_df = concept_df.set_index("CAV").reindex(cav_order).reset_index()
    sns.barplot(
        x="CAV",
        y="Concept Relevance",
        hue="CAV",
        data=ordered_df,
        order=cav_order,
        hue_order=cav_order,
        errorbar=None,
        ax=ax,
    )
    _add_concept_relevance_errorbars(
        ax,
        {row["CAV"]: float(row["SEM"]) for _, row in ordered_df.iterrows()},
        cav_order,
    )

    ymax = float((ordered_df["Concept Relevance"] + ordered_df["SEM"]).max())
    vmax = 0.5 if ymax > 0.44 else 0.45
    vmax = max(vmax, ymax + 0.01)
    ymin = 0.25 if float(ordered_df["Concept Relevance"].min()) >= 0.25 else 0.0
    if ymin == 0.25:
        ticks = [0.25, 0.3, 0.35, 0.4, 0.45]
        if vmax >= 0.5:
            ticks.append(0.5)
    else:
        vmax = max(0.05, float(np.ceil(vmax / 0.05) * 0.05))
        ticks = np.arange(ymin, vmax + 1e-9, 0.05).tolist()
    ax.set_ylim(ymin, vmax)
    ax.set_yticks(ticks)

    ax.xaxis.grid(False)
    ax.yaxis.grid(True)
    fig.savefig(save_dir / "concept_relevance_std.png", bbox_inches="tight", dpi=500)
    fig.savefig(save_dir / "concept_relevance_std.pdf", bbox_inches="tight", dpi=500)
    plt.close(fig)


def process_result_dir(result_dir: Path, repo_root: Path, dataset_name: str, device: str) -> None:
    print(f"[plot] {result_dir}")
    for metric_type in ("test_accuracy", "test_fnr"):
        metric_df = build_metric_df(result_dir, metric_type)
        plot_metric_with_sem(metric_df, metric_type, result_dir)

    spec = parse_run_spec(result_dir)
    cfg = build_cfg(repo_root, spec, dataset_name=dataset_name, device=device)
    concept_df = compute_concept_relevance_df(result_dir, cfg)
    plot_concept_relevance_with_sem(concept_df, result_dir)


def main() -> None:
    args = parse_args()
    result_dirs = find_result_dirs(args.results_root)
    if not result_dirs:
        raise SystemExit(f"No result directories found under {args.results_root}.")

    for result_dir in result_dirs:
        process_result_dir(
            result_dir=result_dir,
            repo_root=REPO_ROOT,
            dataset_name=args.dataset_name,
            device=args.device,
        )


if __name__ == "__main__":
    main()
