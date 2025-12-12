import logging
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset
from models import get_fn_model_loader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from utils.metrics import calculate_metrics

log = logging.getLogger(__name__)


def _select_from_mapping(cfg: Any, key: str) -> Any:
    """Traverse a (possibly nested) mapping using dot notation."""
    if isinstance(cfg, DictConfig):
        try:
            return OmegaConf.select(cfg, key)
        except (AttributeError, ValueError):
            pass
    current: Any = cfg
    for part in key.split("."):
        if isinstance(current, DictConfig):
            if part in current:
                current = current[part]
            else:
                return None
        elif isinstance(current, Mapping):
            current = current.get(part)
            if current is None:
                return None
        else:
            return None
    return current


def resolve_checkpoint_path(cfg: DictConfig | Mapping[str, Any], model_name: str, dataset_name: str) -> Path:
    checkpoint_dir = Path(get_original_cwd()) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir / f"checkpoint_{model_name}_{dataset_name}.pth"


def load_base_model(
    cfg: DictConfig,
    num_classes: Any,
    device: torch.device | str,
) -> torch.nn.Module:
    """Instantiate the classification model used for CLArC evaluations."""
    device = torch.device(device)

    pretrained = getattr(cfg.model, "pretrained", True)
    ckpt_path = getattr(cfg.model, "ckpt_path", None)

    if ckpt_path is None:
        fallback_path = resolve_checkpoint_path(cfg, cfg.model.name, cfg.dataset.name)
        if fallback_path.exists():
            ckpt_path = str(fallback_path)
        else:
            log.warning("No checkpoint found at %s. Using model defaults.", fallback_path)

    model_loader = get_fn_model_loader(cfg.model.name)
    model = model_loader(
        n_class=num_classes,
        ckpt_path=ckpt_path,
        pretrained=pretrained,
    ).to(device)
    model.eval()
    return model


def compose_clarc_config(config: DictConfig) -> dict[str, Any]:
    base_cfg = OmegaConf.to_container(config.correction, resolve=True)  # type: ignore[arg-type]
    to_filter = ["method", "dir_precomputed_data", "mode"]
    clarc_cfg = {k: v for k, v in base_cfg.items() if k not in to_filter}    # type: ignore
    clarc_cfg["p_artifact"] = config.dataset.p_artifact
    clarc_cfg["artifact_type"] = config.dataset.artifact_type
    clarc_cfg["lsb_factor"] = getattr(config.dataset, "lsb_factor", None)
    clarc_cfg["use_backdoor_model"] = clarc_cfg.get("use_backdoor_model", None)
    clarc_cfg["dir_precomputed_data"] = config.correction.dir_precomputed_data
    clarc_cfg["mode"] = config.cav.cav_mode
    return clarc_cfg    # type: ignore


def build_clarc_kwargs(config: DictConfig, dataset: Dataset) -> dict[str, Any]:
    idxs_train, _, _ = dataset.do_train_val_test_split(  # type: ignore
        val_split=config.train.val_ratio,
        test_split=config.train.test_ratio,
        seed=config.train.random_seed
    )
    return {
        "artifact_sample_ids": dataset.sample_ids_by_artifact[config.dataset['artifact']],   # type: ignore
        "sample_ids": idxs_train,   # type: ignore
        "classes": dataset.classes,  # type: ignore
        "eval_mode": True,
    }

def plot_concept_similarities(df_similarities, save_dir):
    interesting_concepts = [
        "timestamp", 
        "box", 
        "Bangs", 
        "Blond_Hair", 
        "Wearing_Necklace", 
        "Pointy_Nose",
        "Rosy_Cheeks", 
        # "High_Cheekbones", 
        # "Smiling", 
        # "Black_Hair", 
        # "Young", 
        # "Wearing_Necklace", 
        # "High_Cheekbones"
    ]

    fig = plt.figure(figsize=(8, 3))
    plt.axhline(0, color='black', linewidth=1, alpha=0.5, linestyle='--')  # Add this line for the horizontal line

    sns.boxplot(df_similarities.loc[df_similarities["concept"].isin(interesting_concepts)], x="concept", y="v", hue="model")
    plt.xticks(rotation=30)
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')  
    plt.ylabel("Cos. sim. CAV/acts (w concept)")
    [fig.savefig(save_dir / f"concept_similarities.{ending}", bbox_inches="tight") for ending in ["png", "pdf"]]
    plt.close()

def plot_confusion_matrices(confusion_matrices, save_dir, model_tag: str = ""):
    metrics_attacked = calculate_metrics(confusion_matrices['attacked'])
    metrics_clean = calculate_metrics(confusion_matrices['clean'])

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    sns.heatmap(confusion_matrices['attacked'], 
                annot=True, fmt='d', cmap='Blues', cbar=True, ax=axes[0],
                xticklabels=['Others', 'Blonde'],
                yticklabels=['Others', 'Blonde'])
    axes[0].set_title(f'Confusion Matrix (Attacked)\n'
                    f'Accuracy: {metrics_attacked[0]:.2f}\n'
                    f'False Positive Rate: {metrics_attacked[1]:.2f}\n'
                    f'False Negative Rate: {metrics_attacked[2]:.2f}\n'
                    f'Recall: {metrics_attacked[3]:.2f}\n'
                    f'Precision: {metrics_attacked[4]:.2f}')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')

    sns.heatmap(confusion_matrices['clean'], 
                annot=True, fmt='d', cmap='Blues', cbar=True, ax=axes[1],
                xticklabels=['Others', 'Blonde'],
                yticklabels=['Others', 'Blonde'])
    axes[1].set_title(f'Confusion Matrix (Clean)\n'
                    f'Accuracy: {metrics_clean[0]:.2f}\n'
                    f'False Positive Rate: {metrics_clean[1]:.2f}\n'
                    f'False Negative Rate: {metrics_clean[2]:.2f}\n'
                    f'Recall: {metrics_clean[3]:.2f}\n'
                    f'Precision: {metrics_clean[4]:.2f}')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')

    plt.tight_layout()
    filename = "confusion_matrices" if not model_tag else f"confusion_matrices_{model_tag}"
    [fig.savefig(save_dir / f"{filename}.{ending}", bbox_inches="tight") for ending in ["png", "pdf"]]
    plt.close()

def plot_metric_comparison(df, save_dir):
    plt.rcParams.update({'font.size': 9, 'legend.fontsize': 9, 'axes.titlesize': 11})

    metric_names = {
        "test_accuracy": "Accuracy",
        "test_fnr": "False Positive Rate",
    }
    for metric_type in ["test_accuracy", "test_fnr"]:
        sns.set_style("whitegrid")
        fig = plt.figure(figsize=(6, 4))
        sns.barplot(x='Category', y='Value', hue='Model', 
                    data=df[df["Metric Type"] == metric_type])
        plt.ylabel(metric_names[metric_type])
        plt.xlabel("")
        if metric_type == "test_accuracy":
            plt.ylim(0.5, 0.95)
            plt.yticks([0.5, 0.6, 0.7, 0.8, 0.9])
        # plt.title("")
        # plt.title(metric_names[metric_type])
        plt.legend(title='', loc='upper left', bbox_to_anchor=(-.22, -.15),ncols=3)
        [fig.savefig(save_dir / f"metric_comparison_{metric_type}.{ending}", bbox_inches="tight", dpi=500) for ending in ["png", "pdf"]]
        plt.close()