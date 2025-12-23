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
from experiments.model_correction import get_correction_method
from experiments.model_correction.evaluate_by_subset_attacked import evaluate_by_subset_attacked
from experiments.model_correction.utils import load_base_model, compose_clarc_config, build_clarc_kwargs, plot_concept_similarities, plot_metric_comparison, plot_confusion_matrices
from experiments.utils.activations import extract_latents
from experiments.utils.utils import get_save_dir
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


def build_model(config: DictConfig, device: torch.device, dataset: Dataset, cav: torch.Tensor | None, clarc: bool = False):
    num_classes = dataset.get_num_classes() # type: ignore
    model = load_base_model(config, num_classes, device)
    if not clarc:
        return model

    method = get_correction_method(config.correction.method)
    clarc_config = compose_clarc_config(config)
    clarc_kwargs = build_clarc_kwargs(config, dataset)
    correction_model = method(model, clarc_config, cav, **clarc_kwargs)  # type: ignore
    correction_model = correction_model.to(device)
    correction_model.eval()
    return correction_model

def get_activation(module, input_, output_):
            global activations
            activations = output_
            return output_.clone()

def get_activations_ds(model, dataset, config, device, split):
    idxs_train, idxs_val, idxs_test = dataset.do_train_val_test_split(  # type: ignore
        val_split=config.train.val_ratio,
        test_split=config.train.test_ratio,
        seed=config.train.random_seed
    )
    sets = {
        'train': idxs_train,
        'val': idxs_val,
        'test': idxs_test,
    }

    dataset_split = dataset.get_subset_by_idxs(sets[split])
    dl_split = DataLoader(dataset_split, batch_size=config.train.batch_size, shuffle=False)

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



def evaluate_model_correction(cfg: DictConfig, cav_model: nn.Module, base_model: nn.Module) -> None:
    """Evaluate the model correction using the trained CAVs.
    
    Args:
        cfg (DictConfig): Configuration for the experiment.
        cav_model: Orthogonalized CAV Model.
        base_model: The original base model without correction.
        
    Returns:
        None
    """
    save_dir = get_save_dir(cfg)
    results_dir = save_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    log.info("Seeding RNGs with %s", cfg.train.random_seed)
    seed_everything(int(cfg.train.random_seed))

    device = torch.device(cfg.train.device)
    log.info(f"Using device {device}")

    log.info('Initializing dataset')
    dataset = instantiate(cfg.dataset)
    concepts = dataset.get_concept_names()
    artifact_id = concepts.index(cfg.dataset.artifact)

    log.info('Building components')
    base_cavs, _ = base_model.get_params(normalize=True)  # type: ignore
    orth_cavs, _ = cav_model.get_params(normalize=True)   # type: ignore
    model_vanilla = build_model(cfg, device, dataset=dataset, cav=None)
    model_baseline = build_model(cfg, device, dataset=dataset, cav=base_cavs[artifact_id, :], clarc=True)    
    model_orth = build_model(cfg, device, dataset=dataset, cav=orth_cavs[artifact_id, :], clarc=True)

    log.info("Extracting activations on test set.")
    activations_vanilla, _ = get_activations_ds(model_vanilla, dataset, cfg, device, split="test")
    activations_clarc, metadata_concepts = get_activations_ds(model_baseline, dataset, cfg, device, split="test")
    activations_clarc_orth, metadata_concepts = get_activations_ds(model_orth, dataset, cfg, device, split="test")

    log.info("Computing cosine similarities.")
    similarities_vanilla = {cname: cosine_similarities_batch(activations_vanilla[metadata_concepts[cname] == 1], base_cavs[i, :] ) for i, cname in enumerate(concepts)} # type: ignore
    similarities_baseline = {cname: cosine_similarities_batch(activations_clarc[metadata_concepts[cname] == 1], base_cavs[i, :] ) for i, cname in enumerate(concepts)} # type: ignore
    similarities_orth = {cname: cosine_similarities_batch(activations_clarc_orth[metadata_concepts[cname] == 1], orth_cavs[i, :] ) for i, cname in enumerate(concepts)} # type: ignore

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
    plot_concept_similarities(df_similarities, results_dir)
    df_similarities.to_pickle(results_dir / "concept_similarities.pkl")

    log.info("Evaluating by subset attacked.")
    metrics_vanilla, cm_vanilla = evaluate_by_subset_attacked(cfg, model_vanilla, dataset, return_cm=True)
    metrics_baseline, cm_baseline = evaluate_by_subset_attacked(cfg, model_baseline, dataset,  return_cm=True)
    metrics_orth, cm_orth = evaluate_by_subset_attacked(cfg, model_orth, dataset, return_cm=True) 

    data = []
    for metric_name, value in metrics_vanilla.items():
        data.append({'Metric': metric_name, 'Value': value, 'Model': 'Vanilla'})
    for metric_name, value in metrics_baseline.items():
        data.append({'Metric': metric_name, 'Value': value, 'Model': 'Baseline CAV'})
    for metric_name, value in metrics_orth.items():
        data.append({'Metric': metric_name, 'Value': value, 'Model': 'Orthogonal CAV'})
    df = pd.DataFrame(data)

    selected_metrics = [
        f"test_accuracy_{split}" for split in ("ch", "attacked", "clean")
    ] + [
        f"test_fpr_{cls}_{split}"
        for cls in dataset.classes
        for split in ("ch", "attacked", "clean")
    ] + [
        f"test_fnr_{cls}_{split}"
        for cls in dataset.classes
        for split in ("ch", "attacked", "clean")
    ]

    df_filtered = df[df['Metric'].isin(selected_metrics)]
    df_filtered['Category'] = df_filtered['Metric'].str.extract(r'_(clean|attacked|ch)')[0]
    df_filtered['Metric Type'] = df_filtered['Metric'].str.replace(r'_(1|ch|attacked|clean)', '', regex=True)
    df_filtered = df_filtered.loc[~(df_filtered['Category'] == "ch")]
    plot_metric_comparison(df_filtered, results_dir)
    df.to_pickle(results_dir / "all_metrics.pkl")
    df_filtered.to_pickle(results_dir / "selected_metrics.pkl")

    metrics_payload = {
        "vanilla": metrics_vanilla,
        "baseline": metrics_baseline,
        "orthogonal": metrics_orth,
    }
    confusion_payload = {
        "vanilla": cm_vanilla,
        "baseline": cm_baseline,
        "orthogonal": cm_orth,
    }
    with open(results_dir / "metrics_per_model.pkl", "wb") as f:
        pickle.dump(metrics_payload, f)
    with open(results_dir / "confusion_per_model.pkl", "wb") as f:
        pickle.dump(confusion_payload, f)

    for cm, name in [(cm_vanilla, "vanilla"), (cm_baseline, "baseline"), (cm_orth, "orthogonal")]:
        plot_confusion_matrices(cm["test"], results_dir, model_tag=name)
    
