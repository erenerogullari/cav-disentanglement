import pickle
import random
import logging
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import Dataset, DataLoader

from experiments.clarc import get_correction_method, evaluate_by_subset_attacked
from experiments.clarc.utils import (
    first_config_value,
    get_dataset_name,
    get_model_name,
    load_base_model,
    require_config_value,
)
from experiments.utils.activations import extract_latents
from experiments.utils.utils import name_experiment
from utils.cav import compute_cavs
from utils.distance import cosine_similarities_batch
from utils.metrics import calculate_metrics

log = logging.getLogger(__name__)

def seed_everything(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(config: DictConfig, device: torch.device, dataset: Optional[Dataset] = None, clarc: bool = False):
    model = load_base_model(config, dataset, device)
    if not clarc:
        return model

    if dataset is None:
        raise ValueError("Dataset is required to instantiate CLArC models.")

    method_name = first_config_value(config, ["correction.method", "method"], default=None)
    if method_name is None:
        raise ValueError("Please specify 'correction.method' in the configuration.")
    method = get_correction_method(method_name)
    dataset_name = get_dataset_name(config)
    model_name = get_model_name(config)
    clarc_config = _compose_clarc_config(config, dataset, dataset_name, model_name)
    clarc_kwargs = _build_clarc_kwargs(config, dataset, clarc_config)
    correction_model = method(model, clarc_config, **clarc_kwargs)  # type: ignore
    correction_model = correction_model.to(device)
    correction_model.eval()
    return correction_model


def _compose_clarc_config(
    config: DictConfig,
    dataset: Dataset,
    dataset_name: str,
    model_name: str,
) -> dict[str, Any]:
    base_cfg = {}
    if "correction" in config and config.correction is not None:
        base_cfg = OmegaConf.to_container(config.correction, resolve=True)  # type: ignore[arg-type]
        if not isinstance(base_cfg, dict):
            base_cfg = {}
    clarc_cfg = {k: v for k, v in base_cfg.items() if k != "method"}
    layer_name = clarc_cfg.pop("layer", None) or clarc_cfg.get("layer_name")
    if layer_name is None:
        layer_name = first_config_value(config, ["cav.layer", "correction.layer"], default=None)
    if layer_name is None:
        raise ValueError("Could not determine the layer to apply CLArC to.")
    clarc_cfg["layer_name"] = layer_name
    clarc_cfg["dataset_name"] = clarc_cfg.get("dataset_name", dataset_name)
    if "cav_scope" not in clarc_cfg or clarc_cfg["cav_scope"] is None:
        classes = getattr(dataset, "classes", None)
        if classes is not None:
            clarc_cfg["cav_scope"] = list(range(len(classes)))
    clarc_cfg["model_name"] = clarc_cfg.get("model_name", model_name)
    clarc_cfg["direction_mode"] = clarc_cfg.get("direction_mode") or first_config_value(
        config, ["correction.direction_mode", "direction_mode"], default="signal"
    )
    clarc_cfg["p_artifact"] = clarc_cfg.get("p_artifact") or first_config_value(
        config, ["dataset.p_artifact", "p_artifact"], default=None
    )
    if clarc_cfg["p_artifact"] is None:
        raise ValueError("p_artifact must be provided to build a CLArC model.")
    clarc_cfg["artifact_type"] = clarc_cfg.get("artifact_type") or first_config_value(
        config, ["dataset.artifact_type", "artifact_type"], default=None
    )
    clarc_cfg["lsb_factor"] = clarc_cfg.get("lsb_factor") or first_config_value(
        config, ["dataset.lsb_factor", "lsb_factor"], default=None
    )
    clarc_cfg["use_backdoor_model"] = clarc_cfg.get("use_backdoor_model") or first_config_value(
        config, ["correction.use_backdoor_model", "use_backdoor_model"], default=False
    )
    dir_precomputed = clarc_cfg.get("dir_precomputed_data") or first_config_value(
        config, ["dir_precomputed_data"], default=None
    )
    if dir_precomputed is None:
        raise ValueError("'dir_precomputed_data' is required for CLArC corrections.")
    clarc_cfg["dir_precomputed_data"] = dir_precomputed
    return clarc_cfg


def _build_clarc_kwargs(config: DictConfig, dataset: Dataset, clarc_cfg: dict[str, Any]) -> dict[str, Any]:
    artifact_name = first_config_value(
        config, ["correction.artifact_name", "artifact_name"], default="timestamp"
    )
    artifact_ids = _get_artifact_sample_ids(dataset, artifact_name)
    sample_ids = getattr(dataset, "idxs_train", None)
    if sample_ids is None:
        raise ValueError("Dataset must expose 'idxs_train' for CLArC corrections.")
    mode = clarc_cfg.get("mode") or first_config_value(
        config, ["correction.mode", "correction.cav_mode", "cav.cav_mode"], default="cavs_full"
    )
    return {
        "artifact_sample_ids": np.array(artifact_ids),
        "sample_ids": np.array(sample_ids),
        "mode": mode,
        "classes": getattr(dataset, "classes", None),
        "eval_mode": True,
    }


def _get_artifact_sample_ids(dataset: Dataset, artifact_name: str):
    if hasattr(dataset, "sample_ids_by_artifact"):
        mapping = getattr(dataset, "sample_ids_by_artifact")
        if mapping and artifact_name in mapping:
            return mapping[artifact_name]
    alt_attr = getattr(dataset, f"{artifact_name}_ids", None)
    if alt_attr is not None:
        return alt_attr
    raise ValueError(f"Artifact '{artifact_name}' not available on dataset.")


def get_baseline_cavs(cfg: DictConfig, model: nn.Module, dataset: Dataset) -> torch.Tensor:
    vecs, targets = extract_latents(cfg, model, dataset)
    cavs, _ = compute_cavs(vecs, targets, type=cfg.cav.name, normalize=True)
    return cavs

def get_activation(module, input_, output_):
            global activations
            activations = output_
            return output_.clone()

def get_activations_ds(model, dataset, config, device, split):
    sets = {
        'train': dataset.idxs_train,
        'val': dataset.idxs_val,
        'test': dataset.idxs_test,
    }

    dataset_split = dataset.get_subset_by_idxs(sets[split])
    dl_split = DataLoader(dataset_split, batch_size=config.experiment.batch_size, shuffle=False)

    handles = []
    layer = config.correction.layer_name
    for n, m in model.named_modules():
        if n.endswith(layer):
            handles.append(m.register_forward_hook(get_activation))

    activations_split = None
    for x, y in dl_split:
        _ = model(x.to(device))
        acts_batch = activations.clone().detach().cpu().flatten(start_dim=2).max(dim=2).values
        activations_split = acts_batch if activations_split is None else torch.cat([activations_split, acts_batch])
          
          
    [h.remove() for h in handles]
    return activations_split, dataset_split.metadata

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
    plt.show()
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
    plt.show()
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
        plt.show()
        [fig.savefig(save_dir / f"metric_comparison_{metric_type}.{ending}", bbox_inches="tight", dpi=500) for ending in ["png", "pdf"]]
        plt.close()



def evaluate_model_correction(cfg: DictConfig, cav_model: nn.Module) -> None:
    """Evaluate the model correction using the trained CAVs.
    
    Args:
        cfg (DictConfig): Configuration for the experiment.
        cav_model: Orthogonalized CAV Model.
        
    Returns:
        None
    """
    results_dir = Path("results") / "clarc" / name_experiment(cfg)
    results_dir.mkdir(parents=True, exist_ok=True)
    media_dir = results_dir / "media"
    metrics_dir = results_dir / "metrics"
    media_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    experiment_cfg = cfg.experiment

    log.info("Seeding RNGs with %s", experiment_cfg.seed)
    seed_everything(int(experiment_cfg.seed))

    device = torch.device(experiment_cfg.device)
    log.info(f"Using device {device}")

    log.info('Initializing dataset')
    dataset = instantiate(cfg.dataset)
    concepts = dataset.get_concept_names()
    timestamp_id = concepts.index("timestamp")

    log.info('Building components')
    model_vanilla = build_model(cfg, device, dataset=dataset)
    model_baseline = build_model(cfg, device, dataset=dataset, clarc=True)
    model_orth = build_model(cfg, device, dataset=dataset, clarc=True)
    cavs_baseline = get_baseline_cavs(cfg, model_vanilla, dataset)
    model_baseline.cav = cavs_baseline[timestamp_id, :] 
    cavs_orth = cav_model.get_params()[0]   # type: ignore
    model_orth.cav = cavs_orth[timestamp_id, :] 

    log.info("Extracting activations on test set.")
    activations_vanilla, _ = get_activations_ds(model_vanilla, dataset, cfg, device, split="test")
    activations_clarc, metadata_concepts = get_activations_ds(model_baseline, dataset, cfg, device, split="test")
    activations_clarc_orth, metadata_concepts = get_activations_ds(model_orth, dataset, cfg, device, split="test")

    log.info("Computing cosine similarities.")
    similarities_vanilla = {cname: cosine_similarities_batch(activations_vanilla[metadata_concepts[cname] == 1], cavs_baseline[i, :] ) for i, cname in enumerate(concepts)} # type: ignore
    similarities_baseline = {cname: cosine_similarities_batch(activations_clarc[metadata_concepts[cname] == 1], cavs_baseline[i, :] ) for i, cname in enumerate(concepts)} # type: ignore
    similarities_orth = {cname: cosine_similarities_batch(activations_clarc[metadata_concepts[cname] == 1], activations_clarc_orth[i, :] ) for i, cname in enumerate(concepts)} # type: ignore

    data_similarities = []
    for c, sims in similarities_vanilla.items():
        for v in sims:
            data_similarities.append({'concept': c, 'v': v.item(), 'model': 'Vanilla'})
    for c, sims in similarities_baseline.items():
        for v in sims:
            data_similarities.append({'concept': c, 'v': v.item(), 'model': 'Baseline CAV'})
    for c, sims in similarities_orth.items():
        for v in sims:
            data_similarities.append({'concept': c, 'v': v.item(), 'model': 'Orthogonal CAV'})

    df_similarities = pd.DataFrame(data_similarities)
    plot_concept_similarities(df_similarities, media_dir)
    df_similarities.to_pickle(metrics_dir / "concept_similarities.pkl")

    log.info("Evaluating by subset attacked.")
    accuracy_metrics_vanilla, cm_vanilla = evaluate_by_subset_attacked(cfg, model_vanilla, dataset, return_cm=True)
    accuracy_metrics_baseline, cm_baseline = evaluate_by_subset_attacked(cfg, model_baseline, dataset,  return_cm=True)
    accuracy_metrics_orth, cm_orth = evaluate_by_subset_attacked(cfg, model_orth, dataset, return_cm=True) 

    data = []
    for metric_name, value in accuracy_metrics_vanilla.items():
        data.append({'Metric': metric_name, 'Value': value, 'Model': 'Vanilla'})
    for metric_name, value in accuracy_metrics_baseline.items():
        data.append({'Metric': metric_name, 'Value': value, 'Model': 'Baseline CAV'})
    for metric_name, value in accuracy_metrics_orth.items():
        data.append({'Metric': metric_name, 'Value': value, 'Model': 'Orthogonal CAV'})
    df = pd.DataFrame(data)

    selected_metrics = [
        'test_accuracy_ch', 'test_accuracy_attacked', 'test_accuracy_clean',
        'test_fpr_1_ch', 'test_fpr_1_attacked', 'test_fpr_1_clean',
        'test_fnr_1_ch', 'test_fnr_1_attacked', 'test_fnr_1_clean'
    ]

    df_filtered = df[df['Metric'].isin(selected_metrics)]
    df_filtered['Category'] = df_filtered['Metric'].str.extract(r'_(clean|attacked|ch)')[0]
    df_filtered['Metric Type'] = df_filtered['Metric'].str.replace(r'_(1|ch|attacked|clean)', '', regex=True)
    df_filtered = df_filtered.loc[~(df_filtered['Category'] == "ch")]
    plot_metric_comparison(df_filtered, media_dir)
    df.to_pickle(metrics_dir / "all_metrics.pkl")
    df_filtered.to_pickle(metrics_dir / "selected_metrics.pkl")

    metrics_payload = {
        "vanilla": accuracy_metrics_vanilla,
        "baseline": accuracy_metrics_baseline,
        "orthogonal": accuracy_metrics_orth,
    }
    confusion_payload = {
        "vanilla": cm_vanilla,
        "baseline": cm_baseline,
        "orthogonal": cm_orth,
    }
    with open(metrics_dir / "accuracy_metrics.pkl", "wb") as f:
        pickle.dump(metrics_payload, f)
    with open(metrics_dir / "confusion_matrices.pkl", "wb") as f:
        pickle.dump(confusion_payload, f)

    for cm, name in [(cm_vanilla, "vanilla"), (cm_baseline, "baseline"), (cm_orth, "orthogonal")]:
        plot_confusion_matrices(cm["test"], media_dir, model_tag=name)
    
