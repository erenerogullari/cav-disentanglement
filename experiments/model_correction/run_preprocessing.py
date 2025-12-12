import torch
import os
import h5py
import gc
import numpy as np
import logging
from pathlib import Path
from tqdm import tqdm
import yaml
import random
from torch.utils.data import Dataset
from crp.attribution import CondAttribution
from crp.concepts import ChannelConcept
from zennit.composites import EpsilonPlusFlat
from models import get_canonizer, get_fn_model_loader
from experiments.model_correction.utils import load_base_model
from experiments.utils.activations import extract_latents
from models import TRANSFORMER_MODELS, MODELS_1D
from omegaconf import DictConfig
from typing import Tuple, Optional
from hydra.utils import instantiate

log = logging.getLogger(__name__)

def seed_everything(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_preprocessing(config: DictConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    device = torch.device(config.train.device)
    cache_dir = Path(config.correction.dir_precomputed_data)
    cache_dir.mkdir(parents=True, exist_ok=True)

    seed = config.train.random_seed
    seed_everything(seed)
    dataset = instantiate(config.dataset)
    num_classes = len(dataset.classes)
    model = load_base_model(config, num_classes, device)

    activations, labels = extract_latents(config, model, dataset)
    return activations, labels
