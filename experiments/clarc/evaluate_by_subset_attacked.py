from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import logging
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from datasets import get_dataset, get_dataset_kwargs
from datasets.celeba.artificial_artifact import get_artifact_kwargs
from experiments.clarc import get_correction_method
from utils.metrics import get_accuracy, get_f1, get_auc_label, get_fnr_label, get_fpr_label

from sklearn.metrics import confusion_matrix

log = logging.getLogger(__name__)

def _legacy_config_view(config: DictConfig) -> dict:
    cfg_dict = OmegaConf.to_container(config, resolve=True)
    if not isinstance(cfg_dict, dict):
        return {}
    dataset_cfg = cfg_dict.get("dataset", {}) or {}
    model_cfg = cfg_dict.get("model", {}) or {}
    legacy_cfg = {**cfg_dict, **dataset_cfg}
    if "dataset_name" not in legacy_cfg and isinstance(dataset_cfg, dict):
        dataset_name = dataset_cfg.get("name")
        if dataset_name is not None:
            legacy_cfg["dataset_name"] = dataset_name
    if "model_name" not in legacy_cfg and isinstance(model_cfg, dict):
        model_name = model_cfg.get("name")
        if model_name is not None:
            legacy_cfg["model_name"] = model_name
    return legacy_cfg

def compute_model_scores(
        model: torch.nn.Module,
        dl: torch.utils.data.DataLoader,
        device: str,
        limit_batches: int | None = None):
    model.to(device).eval()
    model_outs = []
    ys = []
    log.info("Computing model scores")
    for i, (x_batch, y_batch) in enumerate(tqdm(dl)):
        if limit_batches and limit_batches == i:
            break
        model_out = model(x_batch.to(device)).detach().cpu()
        model_outs.append(model_out)
        ys.append(y_batch)

    model_outs = torch.cat(model_outs)
    y_true = torch.cat(ys)

    return model_outs, y_true

def compute_metrics(model_outs, y_true, classes=None, prefix="", suffix="", return_cm=False):
    accuracy, standard_err = get_accuracy(model_outs, y_true, se=True)
    results = {
        f"{prefix}accuracy{suffix}": accuracy,
        f"{prefix}accuracy_standard_err{suffix}": standard_err,
        f"{prefix}f1{suffix}": get_f1(model_outs, y_true)
    }

    cm = confusion_matrix(y_true, model_outs.argmax(1))

    if classes is not None:
        results_auc = {f"{prefix}AUC_{classes[class_id]}{suffix}": get_auc_label(y_true, model_outs, class_id)
                       for class_id in range(model_outs.shape[1])}
        
        if y_true.ndim == 1:
            ## single label
            model_preds = torch.eye(len(classes))[model_outs.argmax(1)]
            y_true = torch.eye(len(classes))[y_true]
        else:
            ## multi-label
            model_preds = (torch.sigmoid(model_outs) > .5).type(torch.uint8)

        results_fpr = {f"{prefix}fpr_{classes[class_id]}{suffix}": get_fpr_label(y_true, model_preds, class_id)
                       for class_id in range(model_preds.shape[1]) if y_true[:, class_id].sum() > 0}
        
        results_fnr = {f"{prefix}fnr_{classes[class_id]}{suffix}": get_fnr_label(y_true, model_preds, class_id)
                       for class_id in range(model_preds.shape[1]) if y_true[:, class_id].sum() > 0}
        
        results_acc = {f"{prefix}acc_{classes[class_id]}{suffix}": (y_true[:, class_id] == model_preds[:, class_id]).numpy().mean()
                       for class_id in range(model_preds.shape[1]) if y_true[:, class_id].sum() > 0}
        
        
        results = {**results, **results_auc, **results_fpr, **results_fnr, **results_acc}
    
    if return_cm:
        return results, cm
    return results

def evaluate_by_subset_attacked(config: DictConfig, model: nn.Module, dataset: Dataset, return_cm=False):
    """ Run evaluations for each data split (train/val/test) on 3 variants of datasets:
            1. Same as training (one attacked class)
            2. Attacked (artifact in all classes)
            3. Clean (no artifacts)

    Args:
        config (dict): config for model correction run
    """

    legacy_cfg = _legacy_config_view(config)
    device = config.experiment.device
    dataset_name = legacy_cfg.get("dataset_name", config.dataset.name)

    data_paths = legacy_cfg.get("data_paths", config.dataset.data_paths)
    batch_size = config.experiment.batch_size
    img_size = legacy_cfg.get("image_size", config.dataset.image_size)
    artifact_type = legacy_cfg.get("artifact_type", getattr(config.dataset, "artifact_type", None))
    binary_target = legacy_cfg.get("binary_target", getattr(config.dataset, "binary_target", None))
    artifact_kwargs = get_artifact_kwargs(legacy_cfg)
    dataset_specific_kwargs = get_dataset_kwargs(legacy_cfg)   
        
    sets = {
        'train': dataset.idxs_train,    # type: ignore
        'val': dataset.idxs_val,        # type: ignore
        'test': dataset.idxs_test,      # type: ignore
    }

    dataset_clean = get_dataset(dataset_name)(data_paths=data_paths,
                                              normalize_data=True,
                                              attacked_classes=[],
                                              binary_target=binary_target,
                                              image_size=img_size,
                                              **artifact_kwargs, **dataset_specific_kwargs)

    if "imagenet" in dataset_name:
        all_classes = list(dataset.label_map.keys())    # type: ignore
        if legacy_cfg.get("subset_correction", False):
            sets['test'] = sets['test'][::10]
            sets['val'] = sets['val'][::10]
    else:
        all_classes = dataset.classes if "isic" in dataset_name else range(len(dataset.classes))  # type: ignore

    dataset_attacked = get_dataset(dataset_name)(data_paths=data_paths,
                                                 normalize_data=True,
                                                 p_artifact=1.0,
                                                 image_size=img_size,
                                                 artifact_type=artifact_type,
                                                 binary_target=binary_target,
                                                 attacked_classes=all_classes,
                                                 **artifact_kwargs, **dataset_specific_kwargs)
    metrics_all = {}
    cms_all = {}
    for split in [
        'test', 
        # 'val'
        ]:
        split_set = sets[split]

        dataset_ch_split = dataset.get_subset_by_idxs(split_set)    # type: ignore
        dataset_clean_split = dataset_clean.get_subset_by_idxs(split_set)
        dataset_attacked_split = dataset_attacked.get_subset_by_idxs(split_set)

        dl_clean = DataLoader(dataset_clean_split, batch_size=batch_size, shuffle=False)
        model_outs_clean, y_true_clean = compute_model_scores(model, dl_clean, device)

        dl = DataLoader(dataset_ch_split, batch_size=batch_size, shuffle=False)
        model_outs, y_true = compute_model_scores(model, dl, device)

        dl_attacked = DataLoader(dataset_attacked_split, batch_size=batch_size, shuffle=False)
        model_outs_attacked, y_true_attacked = compute_model_scores(model, dl_attacked, device)

        

        classes = dataset.classes   # type: ignore

        metrics = compute_metrics(model_outs, y_true, classes, prefix=f"{split}_", suffix=f"_ch", return_cm=return_cm)

        metrics_attacked = compute_metrics(model_outs_attacked, y_true_attacked, classes,
                                           prefix=f"{split}_", suffix=f"_attacked", return_cm=return_cm)
        metrics_clean = compute_metrics(model_outs_clean, y_true_clean, classes, prefix=f"{split}_",
                                        suffix=f"_clean", return_cm=return_cm)

        if return_cm:
            cms_all[split] = {
                "attacked": metrics_attacked[1],    # type: ignore
                "ch": metrics[1],   # type: ignore
                "clean": metrics_clean[1],  # type: ignore
            }
            metrics, metrics_attacked, metrics_clean = metrics[0], metrics_attacked[0], metrics_clean[0]    # type: ignore

        metrics_all_split = {**metrics, **metrics_attacked, **metrics_clean}    # type: ignore
        metrics_all = {**metrics_all, **metrics_all_split}
    if return_cm:
        return metrics_all, cms_all
    return metrics_all
