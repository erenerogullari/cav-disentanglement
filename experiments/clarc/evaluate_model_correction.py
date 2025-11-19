from typing import Any
from omegaconf import DictConfig
from hydra.utils import instantiate
import torch
from torch import nn
from typing import Optional
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from experiments.clarc import get_correction_method

def seed_everything(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(config: DictConfig, device: torch.device, clarc: bool = False):
    model = instantiate(config.model)
    if clarc:
        method = get_correction_method(config.correction.method)
        # TODO: Call method with correct args
        # kwargs = {}
        # model = method(model, **kwargs)
    model = model.to(device)
    model.eval()
    return model


def build_dataloader(config: DictConfig) -> DataLoader:
    dataset = instantiate(config.dataset)
    return DataLoader(
        dataset,
        batch_size=config.experiment.batch_size,
        shuffle=False,
        num_workers=config.experiment.num_workers,
        drop_last=False,
    )

def evaluate_model_correction(cfg: DictConfig, cav_model: nn.Module) -> None:
    """Evaluate the model correction using the trained CAVs.
    
    Args:
        cfg (DictConfig): Configuration for the experiment.
        cav_model: The trained CAV model.
        
    Returns:
        None
    """
    # Implement the evaluation logic here
    pass