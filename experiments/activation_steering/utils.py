import os
from omegaconf import DictConfig
from pathlib import Path
import shutil
import logging

from experiments.utils.utils import (
    format_orthogonality_config_name,
    get_target_concepts,
)

log = logging.getLogger(__name__)


def get_move_target_name(cfg: DictConfig) -> str:
    return str(cfg.move_encs.target_concept)


def get_dir_model_config_name(cfg: DictConfig) -> str:
    return format_orthogonality_config_name(
        cfg.dir_model.alpha,
        cfg.dir_model.get("beta", None),
        get_target_concepts(cfg.dir_model),
    )


def get_moved_encodings_root(cfg: DictConfig) -> Path:
    return (
        Path(cfg.experiment.out)
        / "moved_encs"
        / get_move_target_name(cfg)
        / str(cfg.dir_model.name)
        / get_dir_model_config_name(cfg)
    )


def get_decoding_output_root(cfg: DictConfig) -> Path:
    return (
        Path(cfg.experiment.out)
        / "decodings"
        / get_move_target_name(cfg)
        / str(cfg.dir_model.name)
    )


def clean_up(cfg: DictConfig) -> None:
    """Cleans up temporary files created during the experiment.
    Args:
        cfg (DictConfig): Configuration object containing all parameters.
    Returns:
        None  
    """

    cache_dir = Path(cfg.experiment.out) / "moved_encs"
    if os.path.exists(cache_dir):
        log.info(f"Removing all files in {cache_dir}")
        shutil.rmtree(cache_dir)