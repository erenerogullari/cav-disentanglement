import inspect
from pathlib import Path
from typing import Any, Mapping

import torch
from hydra.utils import get_class, instantiate
from omegaconf import DictConfig, OmegaConf

RUNTIME_CAV_KEYS = {
    "_target_",
    "name",
    "layer",
    "cav_mode",
    "alpha",
    "beta",
    "n_targets",
    "optimal_init",
    "exit_criterion",
}


def _to_plain_mapping(cav_cfg: DictConfig | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(cav_cfg, DictConfig):
        raw_cfg = OmegaConf.to_container(cav_cfg, resolve=True)
    else:
        raw_cfg = dict(cav_cfg)
    if not isinstance(raw_cfg, dict):
        raise TypeError("CAV config must resolve to a mapping.")
    return raw_cfg


def build_cav_model_cfg(cav_cfg: DictConfig | Mapping[str, Any]) -> dict[str, Any]:
    raw_cfg = _to_plain_mapping(cav_cfg)
    target = raw_cfg.get("_target_")
    if not isinstance(target, str) or not target:
        raise KeyError("CAV config requires a non-empty '_target_' field.")

    model_cls = get_class(target)
    signature = inspect.signature(model_cls.__init__)
    accepts_var_kw = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()
    )

    model_cfg: dict[str, Any] = {"_target_": target}
    for key, value in raw_cfg.items():
        if key in RUNTIME_CAV_KEYS:
            continue
        if accepts_var_kw or key in signature.parameters:
            model_cfg[key] = value
    return model_cfg


def instantiate_cav_model(
    cav_cfg: DictConfig | Mapping[str, Any],
    n_concepts: int,
    n_features: int,
    device: str = "cpu",
):
    model_cfg = build_cav_model_cfg(cav_cfg)
    return instantiate(
        model_cfg,
        n_concepts=n_concepts,
        n_features=n_features,
        device=device,
    )


def validate_precomputed_g_sae_cache(cache_path: str | Path) -> None:
    path = Path(cache_path)
    if not path.exists():
        raise FileNotFoundError(
            f"G_SAE baseline cache not found at '{path}'. "
            "Train and export baseline G_SAE first (variables/{dataset}/{model}/{layer}/G_SAE.pth)."
        )

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid G_SAE cache format at '{path}': expected a dict payload.")

    entries = payload.get("entries")
    if not isinstance(entries, dict):
        raise ValueError(
            f"Invalid G_SAE cache format at '{path}': missing 'entries' mapping."
        )

    for key in ("normalized", "unnormalized"):
        entry = entries.get(key)
        if not isinstance(entry, dict):
            raise ValueError(
                f"Invalid G_SAE cache format at '{path}': missing entries['{key}']."
            )
        if "cavs" not in entry or "bias" not in entry:
            raise ValueError(
                f"Invalid G_SAE cache format at '{path}': entries['{key}'] must contain 'cavs' and 'bias'."
            )
