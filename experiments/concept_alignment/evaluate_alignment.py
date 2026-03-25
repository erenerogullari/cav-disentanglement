import logging
import math
import pickle
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from crp.attribution import CondAttribution
from hydra.utils import instantiate
from omegaconf import DictConfig
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from zennit.composites import EpsilonPlusFlat

from experiments.utils.utils import get_save_dir

log = logging.getLogger(__name__)

MODEL_ORDER = ["Baseline", "Orthogonal"]

try:
    import seaborn as sns
except ModuleNotFoundError:
    sns = None


def seed_everything(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _is_vit_model(model_name: str) -> bool:
    return model_name.startswith("vit")


def _build_composite(cfg: DictConfig):
    if _is_vit_model(cfg.model.name):
        import zennit.rules as z_rules
        from zennit.composites import LayerMapComposite

        return LayerMapComposite(
            [
                (torch.nn.Conv2d, z_rules.Gamma(100)),
                (torch.nn.Linear, z_rules.Gamma(0.1)),
            ]
        )

    from models import get_canonizer

    canonizers = get_canonizer(cfg.model.name)
    return EpsilonPlusFlat(canonizers=canonizers)


def _resolve_eval_indices(cfg: DictConfig, dataset: Any) -> np.ndarray:
    idxs_train, idxs_val, idxs_test = dataset.do_train_val_test_split(  # type: ignore
        val_split=cfg.train.val_ratio,
        test_split=cfg.train.test_ratio,
        seed=cfg.train.random_seed,
    )
    split_map = {
        "train": np.array(idxs_train),
        "val": np.array(idxs_val),
        "test": np.array(idxs_test),
    }
    split_name = cfg.alignment.split
    if split_name not in split_map:
        raise ValueError(
            f"Unknown split '{split_name}'. Use one of {list(split_map.keys())}."
        )

    sample_ids = split_map[split_name]
    if cfg.alignment.max_samples is not None:
        sample_ids = sample_ids[: int(cfg.alignment.max_samples)]
    return sample_ids


def _resolve_artifact_concepts(cfg: DictConfig, dataset: Any) -> list[str]:
    configured = list(cfg.alignment.artifact_concepts)
    if configured:
        return configured
    if not hasattr(dataset, "sample_ids_by_artifact"):
        raise AttributeError(
            "Dataset has no attribute 'sample_ids_by_artifact'. "
            "Please set alignment.artifact_concepts explicitly."
        )
    return list(dataset.sample_ids_by_artifact.keys())  # type: ignore


def _build_artifact_spec_map(dataset: Any) -> dict[str, tuple[str, dict[str, Any]]]:
    if not hasattr(dataset, "sample_ids_by_artifact"):
        return {}

    artifact_keys = list(dataset.sample_ids_by_artifact.keys())  # type: ignore
    spec_map: dict[str, tuple[str, dict[str, Any]]] = {}

    if hasattr(dataset, "art1_type") and artifact_keys:
        spec_map[artifact_keys[0]] = (
            dataset.art1_type,
            dict(getattr(dataset, "art1_kwargs", {})),
        )
    if hasattr(dataset, "art2_type") and len(artifact_keys) > 1:
        spec_map[artifact_keys[1]] = (
            dataset.art2_type,
            dict(getattr(dataset, "art2_kwargs", {})),
        )

    return spec_map


class PairedArtifactDataset(Dataset):
    """Creates clean/attacked pairs from identical source image IDs."""

    def __init__(
        self,
        dataset: Any,
        sample_ids: np.ndarray,
        artifact_type: str,
        artifact_kwargs: dict[str, Any],
    ) -> None:
        if not hasattr(dataset, "path") or not hasattr(dataset, "metadata"):
            raise AttributeError(
                "PairedArtifactDataset expects dataset.path and dataset.metadata."
            )
        if not hasattr(dataset, "add_artifact"):
            raise AttributeError(
                "Dataset must define add_artifact for deterministic insertion."
            )
        if not hasattr(dataset, "transform_resize"):
            raise AttributeError(
                "Dataset must define transform_resize for deterministic preprocessing."
            )
        self.dataset = dataset
        self.sample_ids = np.array(sample_ids)
        self.artifact_type = artifact_type
        self.artifact_kwargs = artifact_kwargs

    def __len__(self) -> int:
        return len(self.sample_ids)

    def _load_base_image(self, idx: int) -> Image.Image:
        image_id = self.dataset.metadata.iloc[idx]["image_id"]
        img_path = Path(self.dataset.path) / "img_align_celeba" / image_id
        with Image.open(img_path) as image_raw:
            image = image_raw.convert("RGB")
        image = self.dataset.transform_resize(image)
        return image

    def _to_tensor(self, image: Image.Image) -> torch.Tensor:
        if self.dataset.transform:
            image = self.dataset.transform(image)
        return image.float()

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        idx = int(self.sample_ids[item])
        clean_img = self._load_base_image(idx)
        attacked_img = clean_img.copy()
        attacked_img, _ = self.dataset.add_artifact(
            attacked_img, idx, self.artifact_type, **self.artifact_kwargs
        )

        clean_tensor = self._to_tensor(clean_img)
        attacked_tensor = self._to_tensor(attacked_img)
        return clean_tensor, attacked_tensor


def _compute_deltas(
    cfg: DictConfig,
    dataset: Any,
    concept_name: str,
    attribution: CondAttribution,
    composite: Any,
    device: torch.device,
    sample_ids: np.ndarray,
    spec_map: dict[str, tuple[str, dict[str, Any]]],
) -> torch.Tensor:
    from experiments.utils.activations import _get_features

    if concept_name not in spec_map:
        raise KeyError(
            f"No artifact insertion spec found for concept '{concept_name}'. "
            f"Available: {list(spec_map.keys())}"
        )
    artifact_type, artifact_kwargs = spec_map[concept_name]
    ds_pairs = PairedArtifactDataset(dataset, sample_ids, artifact_type, artifact_kwargs)
    dataloader = DataLoader(
        ds_pairs,
        batch_size=cfg.alignment.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
    )

    deltas = []
    for clean_batch, attacked_batch in tqdm(
        dataloader, desc=f"Pairs[{concept_name}]", leave=False
    ):
        clean_latents = _get_features(
            clean_batch,
            cfg.cav.layer,
            attribution,
            composite,
            cfg.cav.cav_mode,
            device=device,
        )
        attacked_latents = _get_features(
            attacked_batch,
            cfg.cav.layer,
            attribution,
            composite,
            cfg.cav.cav_mode,
            device=device,
        )
        deltas.append((attacked_latents - clean_latents).detach().cpu())

    if not deltas:
        raise RuntimeError(f"No paired deltas computed for concept '{concept_name}'.")
    return torch.cat(deltas, dim=0)


def _add_grouped_bar_errorbars(
    ax: plt.Axes,
    sem_lookup: dict[tuple[str, str], float],
    concept_order: list[str],
) -> None:
    for model_name, container in zip(MODEL_ORDER, ax.containers[: len(MODEL_ORDER)]):
        for concept_name, patch in zip(concept_order, container):
            sem = sem_lookup.get((concept_name, model_name))
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


def _plot_alignment_summary(df_summary: pd.DataFrame, save_dir: Path) -> None:
    if df_summary.empty:
        return

    concept_order = list(dict.fromkeys(df_summary["concept"].tolist()))
    fig, ax = plt.subplots(figsize=(max(4, 1.7 * len(concept_order)), 4))
    if sns is not None:
        sns.set_style("whitegrid")
        plt.rcParams.update({"font.size": 9, "legend.fontsize": 9, "axes.titlesize": 11})
        sns.barplot(
            data=df_summary,
            x="concept",
            y="cosine_mean",
            hue="model",
            order=concept_order,
            hue_order=MODEL_ORDER,
            errorbar=None,
            ax=ax,
        )
        _add_grouped_bar_errorbars(
            ax,
            {
                (row["concept"], row["model"]): float(row["cosine_sem"])
                for _, row in df_summary.iterrows()
            },
            concept_order,
        )
        plt.xticks(rotation=20)
    else:
        width = 0.38
        x = np.arange(len(concept_order))
        for model_id, model_name in enumerate(MODEL_ORDER):
            subset = (
                df_summary[df_summary["model"] == model_name]
                .set_index("concept")
                .reindex(concept_order)
            )
            means = subset["cosine_mean"].to_numpy(dtype=float)
            sems = subset["cosine_sem"].to_numpy(dtype=float)
            offset = (model_id - 0.5 * (len(MODEL_ORDER) - 1)) * width
            ax.bar(
                x + offset,
                means,
                width=width,
                label=model_name,
                yerr=sems,
                capsize=4,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(concept_order, rotation=20)

    ax.set_ylim(-1.0, 1.0)
    ax.set_xlabel("Artifact concept")
    ax.set_ylabel("Cosine alignment (mean ± SEM)")
    ax.set_title("Concept alignment against true per-sample deltas")
    ax.legend()
    plt.tight_layout()
    for ending in ["png", "pdf"]:
        fig.savefig(save_dir / f"concept_alignment_bar.{ending}", bbox_inches="tight")
    plt.close(fig)


def _plot_alignment_distribution(df_scores: pd.DataFrame, save_dir: Path) -> None:
    if df_scores.empty:
        return

    if sns is None:
        log.warning("Seaborn not available; skipping distribution plot.")
        return

    sns.set_style("whitegrid")
    plt.rcParams.update({"font.size": 9, "legend.fontsize": 9, "axes.titlesize": 11})
    concept_order = list(dict.fromkeys(df_scores["concept"].tolist()))
    fig, ax = plt.subplots(figsize=(max(4, 1.7 * len(concept_order)), 4))
    sns.boxplot(
        data=df_scores,
        x="concept",
        y="cosine",
        hue="model",
        order=concept_order,
        hue_order=MODEL_ORDER,
        showfliers=False,
        ax=ax,
    )
    ax.set_ylim(-1.0, 1.0)
    ax.set_xlabel("Artifact concept")
    ax.set_ylabel("Per-sample cosine alignment")
    ax.set_title("Per-sample concept alignment distribution")
    plt.xticks(rotation=20)
    plt.tight_layout()
    for ending in ["png", "pdf"]:
        fig.savefig(save_dir / f"concept_alignment_box.{ending}", bbox_inches="tight")
    plt.close(fig)


def evaluate_concept_alignment(
    cfg: DictConfig, cav_model: nn.Module, base_cav_model: nn.Module
) -> None:
    from experiments.model_correction.utils import load_base_model

    save_dir = get_save_dir(cfg)
    results_dir = save_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    seed = int(cfg.train.random_seed)
    log.info("Seeding RNGs with %s", seed)
    seed_everything(seed)

    device = torch.device(cfg.train.device)
    dataset = instantiate(cfg.dataset)

    sample_ids = _resolve_eval_indices(cfg, dataset)
    concept_names = dataset.get_concept_names()
    artifact_concepts = _resolve_artifact_concepts(cfg, dataset)
    artifact_spec_map = _build_artifact_spec_map(dataset)

    num_classes = (
        dataset.get_num_classes() if hasattr(dataset, "get_num_classes") else len(dataset.classes)  # type: ignore
    )
    model = load_base_model(cfg, num_classes, device)
    model.eval()
    attribution = CondAttribution(model)
    composite = _build_composite(cfg)

    cavs_baseline, _ = base_cav_model.get_params(normalize=True)  # type: ignore
    cavs_orthogonal, _ = cav_model.get_params(normalize=True)  # type: ignore
    cav_sets = {
        "Baseline": cavs_baseline.cpu(),
        "Orthogonal": cavs_orthogonal.cpu(),
    }

    summary_rows = []
    metrics_payload: dict[str, dict[str, dict[str, float]]] = {}
    per_sample_payload: dict[str, dict[str, np.ndarray]] = {}
    per_sample_rows = []

    log.info(
        "Evaluating concept alignment on split=%s with %d samples.",
        cfg.alignment.split,
        len(sample_ids),
    )
    for concept in artifact_concepts:
        if concept not in concept_names:
            log.warning(
                "Concept '%s' not found in dataset concepts. Skipping. Available: %s",
                concept,
                concept_names,
            )
            continue

        deltas = _compute_deltas(
            cfg=cfg,
            dataset=dataset,
            concept_name=concept,
            attribution=attribution,
            composite=composite,
            device=device,
            sample_ids=sample_ids,
            spec_map=artifact_spec_map,
        )
        concept_id = concept_names.index(concept)
        metrics_payload[concept] = {}
        per_sample_payload[concept] = {}

        for model_name, cavs in cav_sets.items():
            cav = cavs[concept_id].to(deltas)
            cosines = F.cosine_similarity(
                deltas,
                cav.unsqueeze(0).expand_as(deltas),
                dim=1,
                eps=1e-12,
            )
            cos_np = cosines.numpy()

            n = int(cos_np.shape[0])
            cosine_mean = float(cos_np.mean()) if n > 0 else float("nan")
            cosine_std = float(cos_np.std(ddof=0)) if n > 0 else float("nan")
            cosine_sem = (
                float(cosine_std / math.sqrt(n))
                if n > 0 and np.isfinite(cosine_std)
                else float("nan")
            )
            mean_delta = deltas.mean(dim=0)
            cosine_mean_delta = float(
                F.cosine_similarity(
                    mean_delta.unsqueeze(0),
                    cav.unsqueeze(0),
                    dim=1,
                    eps=1e-12,
                ).item()
            )

            row = {
                "concept": concept,
                "model": model_name,
                "n": n,
                "cosine_mean": cosine_mean,
                "cosine_std": cosine_std,
                "cosine_sem": cosine_sem,
                "cosine_mean_delta": cosine_mean_delta,
            }
            summary_rows.append(row)
            metrics_payload[concept][model_name] = row.copy()

            if cfg.alignment.save_raw_scores:
                per_sample_payload[concept][model_name] = cos_np
                for v in cos_np:
                    per_sample_rows.append(
                        {"concept": concept, "model": model_name, "cosine": float(v)}
                    )

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(results_dir / "alignment_summary.csv", index=False)

    with open(results_dir / "alignment_metrics.pkl", "wb") as f:
        pickle.dump(metrics_payload, f)

    with open(results_dir / "alignment_per_sample.pkl", "wb") as f:
        pickle.dump(per_sample_payload, f)

    _plot_alignment_summary(df_summary, results_dir)
    _plot_alignment_distribution(pd.DataFrame(per_sample_rows), results_dir)

    log.info("Saved concept alignment outputs to %s", results_dir)
