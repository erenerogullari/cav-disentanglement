import os
from omegaconf import DictConfig
from pathlib import Path
import shutil
import logging

log = logging.getLogger(__name__)


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