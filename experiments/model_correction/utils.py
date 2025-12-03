import logging
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from models import get_fn_model_loader

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


def first_config_value(cfg: DictConfig | Mapping[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    """Return the first non-None value for the provided keys."""
    for key in keys:
        value = _select_from_mapping(cfg, key)
        if value is not None:
            return value
    return default


def require_config_value(
    cfg: DictConfig | Mapping[str, Any],
    keys: Sequence[str],
    description: str,
) -> Any:
    """Retrieve a required config value, raising a ValueError if missing."""
    value = first_config_value(cfg, keys, default=None)
    if value is None:
        raise ValueError(f"Missing configuration value for {description} (looked for: {keys}).")
    return value


def get_dataset_name(cfg: DictConfig | Mapping[str, Any]) -> str:
    return require_config_value(cfg, ["dataset.name", "dataset_name"], "dataset name")


def get_model_name(cfg: DictConfig | Mapping[str, Any]) -> str:
    return require_config_value(cfg, ["model.name", "model_name"], "model name")


def resolve_checkpoint_path(cfg: DictConfig | Mapping[str, Any], model_name: str, dataset_name: str) -> Path:
    checkpoint_dir = Path(get_original_cwd()) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir / f"checkpoint_{model_name}_{dataset_name}.pth"


def load_base_model(
    cfg: DictConfig | MutableMapping[str, Any],
    dataset: Any,
    device: torch.device | str,
) -> torch.nn.Module:
    """Instantiate the classification model used for CLArC evaluations."""
    device = torch.device(device)
    model_name = get_model_name(cfg)
    dataset_name = get_dataset_name(cfg)

    num_classes = None
    if dataset is not None and hasattr(dataset, "classes"):
        classes = getattr(dataset, "classes")
        if classes is not None:
            num_classes = len(classes)
    if num_classes is None:
        num_classes = first_config_value(cfg, ["model.n_class", "n_class"], default=None)
    if num_classes is None:
        raise ValueError("Number of classes is required to instantiate the model.")

    pretrained = bool(first_config_value(cfg, ["model.pretrained", "pretrained"], default=True))
    ckpt_path = first_config_value(cfg, ["model.ckpt_path", "ckpt_path"], default=None)

    if ckpt_path:
        checkpoint_file = Path(str(ckpt_path))
        if not checkpoint_file.exists():
            log.warning("Checkpoint %s not found. Falling back to default path.", checkpoint_file)
            ckpt_path = None

    if ckpt_path is None:
        fallback_path = resolve_checkpoint_path(cfg, model_name, dataset_name)
        if fallback_path.exists():
            ckpt_path = str(fallback_path)
        else:
            log.warning("No checkpoint found at %s. Using model defaults.", fallback_path)

    model_loader = get_fn_model_loader(model_name)
    model = model_loader(
        n_class=num_classes,
        ckpt_path=ckpt_path,
        pretrained=pretrained,
    ).to(device)
    model.eval()
    return model
