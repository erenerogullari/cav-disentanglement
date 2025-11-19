import torch
import torch.nn as nn
import torchvision.transforms as T
from crp.attribution import CondAttribution
from crp.image import imgify
from zennit.composites import EpsilonPlusFlat
from zennit.torchvision import ResNetCanonizer, VGGCanonizer
from zennit.canonizers import Canonizer
from omegaconf import DictConfig, OmegaConf
import numpy as np
import logging
import random
import os
import hydra
from typing import List
import matplotlib.pyplot as plt
from tqdm import tqdm
from models import get_fn_model_loader, get_canonizer, get_vgg16
from datasets import get_dataset
from utils.visualizations import visualize_heatmaps, visualize_heatmap_pair
from utils.cav import compute_cavs
from experiments.utils.activations import _get_features, extract_latents
from experiments.utils.utils import name_experiment
from hydra.utils import get_original_cwd
from pathlib import Path

log = logging.getLogger(__name__)

def _resolve_checkpoint_path(cfg_model: DictConfig, dataset_name: str) -> Path:
    checkpoint_dir = Path(get_original_cwd()) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"checkpoint_{cfg_model.name}_{dataset_name}.pth"
    return checkpoint_path

def get_localization(cav: torch.Tensor, x: torch.Tensor, model: nn.Module, canonizers: List[Canonizer], layer: str, cav_mode: str = 'full', device: torch.device | str = 'cpu') -> torch.Tensor:
    """Generate heatmaps for input x using the provided CAVs and model.
    Args:
        cav (torch.Tensor): The Concept Activation Vector.
        x (torch.Tensor): Input data for which heatmaps are to be generated.
        model (nn.Module): The neural network model.
        canonizers (List[Canonizer]): Canonizers to adapt the model for attribution.
        layer (str): The layer of the model to be used for heatmap generation.
        cav_mode (str): Mode of CAV, e.g., 'full' or 'max' or 'avg'.
        device (torch.device | str): Device to perform computations on.
    Returns:
        torch.Tensor: Generated heatmaps.
    """
    attribution = CondAttribution(model)
    x = x.detach().to(device)
    activations = _get_features(x.to(device), layer, attribution, canonizers, cav_mode="full", device=device).detach()
    batch, channels, height, width = activations.shape
    activations = activations.to(device)
    cav = cav.to(device)
    
    if cav_mode == 'full':
        init_rel = (activations * cav[..., None, None])
    elif cav_mode == 'max':
        flat = activations.view(batch, channels, -1)
        max_vals, max_idx = flat.max(dim=2)
        rel_flat = torch.zeros_like(flat)
        rel_flat.scatter_(2, max_idx.unsqueeze(-1), (max_vals * cav).unsqueeze(-1))
        init_rel = rel_flat.view_as(activations)
    elif cav_mode == 'avg':
        spatial_size = max(height * width, 1)
        channel_scores = activations.mean(dim=(-1, -2), keepdim=True) * cav[..., None, None]
        init_rel = channel_scores.expand_as(activations) / spatial_size
    else:
        raise ValueError(f"Invalid cav_mode: {cav_mode}. Choose from 'full', 'max', or 'avg'.")

    composite = EpsilonPlusFlat(canonizers)
    attr = attribution(x.to(device).requires_grad_(), [{}], composite, start_layer=layer, init_rel=init_rel.to(device))
    heatmaps = attr.heatmap.detach().cpu()
    return heatmaps


def binarize_heatmaps(heatmaps:torch.Tensor, kernel_size:int=7, sigma:float=8.0, percentile:int=92):
    """Binarize heatmaps using Gaussian smoothing and percentile thresholding.
    Args:
        heatmaps (torch.Tensor): Heatmaps to be binarized.
        kernel_size (int, optional): Size of the Gaussian kernel. Defaults to 7.
        sigma (float, optional): Standard deviation for Gaussian kernel. Defaults to 8.0
        percentile (int, optional): Percentile for thresholding. Defaults to 92.
    Returns:
        torch.Tensor: Binarized heatmaps.
    """
    gaussian = T.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
    heatmaps_binary = []
    for hm in heatmaps:
        hm_smooth = gaussian(hm.clamp(min=0)[None])[0].numpy()
        thresh = np.percentile(hm_smooth, percentile)
        heatmaps_binary.append((hm_smooth > thresh).astype(np.uint8))
    return torch.Tensor(heatmaps_binary).type(torch.uint8)


def localize_concepts(cfg: DictConfig) -> None:
    """Compute concept-level localization heatmaps for a subset of the dataset.
    Args:
        cfg (DictConfig): Configuration object containing dataset parameters.
        cavs (torch.Tensor): CAVs for the concepts.
        dataset (torch.utils.data.Dataset): The dataset to sample from.
    Returns:
        None
    """
    # Set up the device and seed from the config
    device = cfg.localization.device
    log.info(f"Using device: {device}")
    torch.manual_seed(cfg.localization.random_seed)
    random.seed(cfg.localization.random_seed)
    model_name = name_experiment(cfg)
    original_cwd = hydra.utils.get_original_cwd()
    save_dir = os.path.join(original_cwd, "results", "disentangle_cavs", model_name)

    # Load model and dataset 
    log.info(f"Loading model: {cfg.model.name}")
    ckpt_path = _resolve_checkpoint_path(cfg.model, cfg.dataset.name)
    model = get_fn_model_loader(cfg.model.name)(ckpt_path=ckpt_path,
                                                pretrained=cfg.model.pretrained,
                                                n_class=cfg.model.n_class).to(device)

    log.info(f"Loading dataset: {cfg.dataset.name}")
    dataset = get_dataset(cfg.dataset.name)(data_paths=cfg.dataset.data_paths,
                                        normalize_data=cfg.dataset.normalize_data,
                                        image_size=cfg.dataset.image_size)
    concept_names = dataset.get_concept_names()

    # Load CAVs
    log.info(f"Loading CAVs.")
    if not os.path.exists(os.path.join(save_dir, "cavs.pt")):
        raise FileNotFoundError(f"CAVs not found at {os.path.join(save_dir, 'cavs.pt')}. Please train CAVs first.")
    cavs = torch.load(os.path.join(save_dir, "cavs.pt"), weights_only=True)
    x_latent, labels = extract_latents(cfg, model, dataset)
    cavs_original, bias_original = compute_cavs(x_latent, labels, type=cfg.cav.name, normalize=True)

    canonizers = get_canonizer(cfg.model.name)
    os.makedirs(os.path.join(save_dir, 'localization'), exist_ok=True)

    # Generate heatmaps for each concept
    log.info(f"Generating heatmaps for concepts.")
    for concept_id in tqdm(cfg.localization.concept_ids):
        concept_name = concept_names[concept_id]
        n_samples = cfg.localization.n_samples_each
        sample_ids = random.sample(list(dataset.sample_ids_by_concept[concept_name]), n_samples)
        samples = [dataset[i][0].unsqueeze(0) for i in sample_ids]
        samples = torch.vstack(samples)         # Shape [n_samples_each, 3, 224, 224]
        samples.requires_grad_()

        heatmaps_original = get_localization(cavs_original[concept_id].unsqueeze(0), samples, model, canonizers, cfg.cav.layer, cfg.localization.cav_mode, device).unsqueeze(1)   # Shape [n_samples_each, 1, 224, 224]
        heatmaps_disentangled = get_localization(cavs[concept_id].unsqueeze(0), samples, model, canonizers, cfg.cav.layer, cfg.localization.cav_mode, device).unsqueeze(1)                     # Shape [n_samples_each, 1, 224, 224]
        heatmaps = torch.cat([heatmaps_original, heatmaps_disentangled], dim=1)  # Shape [n_samples_each, 2, 224, 224]

        sample_latents = x_latent[sample_ids].to(device=cavs.device, dtype=cavs.dtype)
        dp_original = torch.matmul(sample_latents, cavs_original[concept_id].to(sample_latents))
        dp_disentangled = torch.matmul(sample_latents, cavs[concept_id].to(sample_latents))
        dot_matrix = torch.stack([dp_original, dp_disentangled], dim=1).detach().cpu()

        titles = ["Original Image", "CAVs Original", "CAVs Disentangled"]
        for i in range(n_samples):
            fig = visualize_heatmaps(
                samples[i].detach().cpu(),
                heatmaps[i, :, :, :],
                suptitle=f"Heatmaps for {concept_name}",
                titles=titles,
                dot_products=dot_matrix[i].tolist(),
                display=False,
            )
            fig.savefig(os.path.join(save_dir, 'localization', f"{concept_id}_{concept_name}_sample{i}.png"))

    log.info(f"Localization completed. Results saved to {os.path.join(save_dir, 'localization')}.")


def colocalize_concept_pairs(cfg: DictConfig) -> None:
    """Compute concept-level colocalization heatmaps for a subset of the dataset.
    Args:
        cfg (DictConfig): Configuration object containing dataset parameters.
    Returns:
        None
    """
    # Set up the device and seed from the config
    device = cfg.localization.device
    log.info(f"Using device: {device}")
    torch.manual_seed(cfg.localization.random_seed)
    random.seed(cfg.localization.random_seed)
    model_name = name_experiment(cfg)
    original_cwd = hydra.utils.get_original_cwd()
    save_dir = os.path.join(original_cwd, "results", "disentangle_cavs", model_name)

    # Load model and dataset 
    log.info(f"Loading model: {cfg.model.name}")
    ckpt_path = _resolve_checkpoint_path(cfg.model, cfg.dataset.name)
    model = get_fn_model_loader(cfg.model.name)(ckpt_path=ckpt_path,
                                                pretrained=cfg.model.pretrained,
                                                n_class=cfg.model.n_class).to(device)

    log.info(f"Loading dataset: {cfg.dataset.name}")
    dataset = get_dataset(cfg.dataset.name)(data_paths=cfg.dataset.data_paths,
                                        normalize_data=cfg.dataset.normalize_data,
                                        image_size=cfg.dataset.image_size)
    concept_names = dataset.get_concept_names()

    # Load CAVs
    log.info(f"Loading CAVs.")
    if not os.path.exists(os.path.join(save_dir, "cavs.pt")):
        raise FileNotFoundError(f"CAVs not found at {os.path.join(save_dir, 'cavs.pt')}. Please train CAVs first.")
    cavs = torch.load(os.path.join(save_dir, "cavs.pt"), weights_only=True)
    x_latent, labels = extract_latents(cfg, model, dataset)
    cavs_original, bias_original = compute_cavs(x_latent, labels, type=cfg.cav.name, normalize=True)

    canonizers = get_canonizer(cfg.model.name)
    os.makedirs(os.path.join(save_dir, 'colocalization'), exist_ok=True)

    # Generate heatmaps for each concept pair
    log.info(f"Generating heatmaps for concept pairs.")
    for concept_pair in tqdm(cfg.localization.concept_pair_ids):
        concept_name1 = concept_names[concept_pair[0]]
        concept_name2 = concept_names[concept_pair[1]]
        n_samples = cfg.localization.n_samples_each
        samples1 = dataset.sample_ids_by_concept[concept_name1]
        samples2 = dataset.sample_ids_by_concept[concept_name2]

        # Get the intersection
        samples = list(set(samples1).intersection(set(samples2)))
        if len(samples) == 0:
            log.warning(f"No samples found for concept pair ({concept_name1}, {concept_name2}). Skipping...")
            continue
        sample_ids = random.sample(samples, min(n_samples, len(samples)))
        samples = [dataset[i][0].unsqueeze(0) for i in sample_ids]
        samples = torch.vstack(samples)         # Shape [n_samples_each, 3, 224, 224]
        samples.requires_grad_()

        heatmaps_original_1 = get_localization(cavs_original[concept_pair[0]].unsqueeze(0), samples, model, canonizers, cfg.cav.layer, cfg.localization.cav_mode, device).unsqueeze(1)   # Shape [n_samples_each, 1, 224, 224]
        heatmaps_original_2 = get_localization(cavs_original[concept_pair[1]].unsqueeze(0), samples, model, canonizers, cfg.cav.layer, cfg.localization.cav_mode, device).unsqueeze(1)   # Shape [n_samples_each, 1, 224, 224]
        heatmaps_disentangled_1 = get_localization(cavs[concept_pair[0]].unsqueeze(0), samples, model, canonizers, cfg.cav.layer, cfg.localization.cav_mode, device).unsqueeze(1)                     # Shape [n_samples_each, 1, 224, 224]
        heatmaps_disentangled_2 = get_localization(cavs[concept_pair[1]].unsqueeze(0), samples, model, canonizers, cfg.cav.layer, cfg.localization.cav_mode, device).unsqueeze(1)                     # Shape [n_samples_each, 1, 224, 224]
        heatmaps_original = torch.cat([heatmaps_original_1, heatmaps_original_2], dim=1)          # Shape [n_samples_each, 2, 224, 224]
        heatmaps_disentangled = torch.cat([heatmaps_disentangled_1, heatmaps_disentangled_2], dim=1)  # Shape [n_samples_each, 2, 224, 224]
        sample_latents = x_latent[sample_ids].to(device=cavs.device, dtype=cavs.dtype)
        dp_original = torch.matmul(sample_latents, cavs_original[concept_pair].to(sample_latents).T).detach().cpu()
        dp_disentangled = torch.matmul(sample_latents, cavs[concept_pair].to(sample_latents).T).detach().cpu()

        titles = ["Original Image", concept_name1, concept_name2]
        row_titles = ["Original CAV", "Disentangled CAV"]
        for idx, sample_id in enumerate(sample_ids):
            pair_heatmaps = torch.stack([heatmaps_original[idx], heatmaps_disentangled[idx]]).detach().cpu()
            dot_products = torch.stack([dp_original[idx], dp_disentangled[idx]]).detach().cpu()
            fig = visualize_heatmap_pair(
                samples[idx].detach().cpu(),
                pair_heatmaps,
                titles=titles,
                row_titles=row_titles,
                dot_products=dot_products,
                display=False,
            )
            filename = f"pair_{concept_pair[0]}-{concept_pair[1]}_{concept_name1}_{concept_name2}_sample{idx}.png"
            fig.savefig(os.path.join(save_dir, 'colocalization', filename))

    log.info(f"Colocalization completed. Results saved to {os.path.join(save_dir, 'colocalization')}.")


if __name__ == "__main__":
    # For debugging purposes
    model = get_vgg16("checkpoints/checkpoint_vgg16_celeba.pth", pretrained=True, n_class=2)
    canonizers = [VGGCanonizer()]
    layer = "features.28"
    device = torch.device("mps")
    model.to(device)
    dataset = get_dataset("celeba")(["/Users/erogullari/datasets/"])
    concepts = dataset.get_concept_names()
    cavs = torch.load("checkpoints/scav_vgg16_celeba.pth", weights_only=True)["weights"].squeeze().to(device)

    # Sample some data points
    n_samples = 5
    sample_ids = random.sample(range(len(dataset)), n_samples)
    samples = [dataset[i][0].unsqueeze(0) for i in sample_ids]
    samples = torch.vstack(samples)         # Shape [n_samples, 3, 224, 224]
    samples.requires_grad_()

    concept_id = 5
    heatmaps = get_localization(cavs[concept_id].unsqueeze(0), samples, model, canonizers, layer, 'max', device).unsqueeze(1) # Shape [n_samples, 224, 224]
    titles = ["Original Image", "Heatmap"]
    for i in range(n_samples):
        fig = visualize_heatmaps(samples[i].detach(), heatmaps[i, :, :, :], subplots_size=(1,2), suptitle=f"Heatmaps for {concepts[concept_id]}")
        fig.savefig(f"results/tmp/heatmaps_{i}.png")
