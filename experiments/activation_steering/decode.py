import random
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torchvision
from omegaconf import DictConfig
from tqdm import tqdm

from hydra.utils import instantiate

import logging
log = logging.getLogger(__name__)


def seed_everything(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(config: DictConfig, device: torch.device):
    model = instantiate(config.model)
    model = model.to(device)
    model.eval()
    return model


def _format_float(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    text = f"{value}"
    if "e" in text or "E" in text:
        return f"{value:.4g}"
    return text.rstrip("0").rstrip(".") if "." in text else text


# def save_image_tensor(img: torch.Tensor, path: Path) -> None:
#     tensor = img.detach().cpu()
#     if tensor.dim() == 4 and tensor.size(0) == 1:
#         tensor = tensor.squeeze(0)
#     if tensor.dim() != 3:
#         raise ValueError("Expected image tensor of shape (C, H, W).")
#     tensor = tensor.clamp(-1.0, 1.0)
#     tensor = (tensor + 1.0) / 2.0
#     torchvision.utils.save_image(tensor, str(path))


def save_batch(
    output_root: Path,
    image_format: str,
    alpha: float,
    step_size: Optional[float],
    batch_indices: torch.Tensor,
    originals: torch.Tensor,
    decodings: torch.Tensor,
) -> None:
    alpha_dir = output_root / f"alpha{_format_float(alpha)}"
    index_list = batch_indices.detach().cpu().tolist()

    for i, img_idx in enumerate(index_list):
        orig_img = originals[i]
        dec_img = decodings[i]
        img_dir = alpha_dir / f"img{int(img_idx)}"
        img_dir.mkdir(parents=True, exist_ok=True)

        original_path = img_dir / f"original.{image_format}"
        if not original_path.exists():
            torchvision.utils.save_image(orig_img, original_path)

        step_suffix = _format_float(step_size) if step_size is not None else "0"
        decoded_path = img_dir / f"step_size{step_suffix}.{image_format}"
        torchvision.utils.save_image(dec_img, decoded_path)


def run_decode(config: DictConfig) -> None:
    experiment_cfg = config.experiment
    alpha = config.dir_model.alpha

    log.info("Seeding RNGs with %s", experiment_cfg.seed)
    seed_everything(int(experiment_cfg.seed))

    device = torch.device(experiment_cfg.device)
    log.info("Using device %s", device)

    move_cfg = config.move_encs
    max_images = getattr(move_cfg, "num_images", 10)
    decode_idxs = list(range(max_images))

    model = build_model(config, device)

    image_format = getattr(experiment_cfg, "format", "png")
    cache_dir = Path(config.experiment.out)
    output_root = cache_dir / "decodings" / config.dir_model.name
    output_root.mkdir(parents=True, exist_ok=True)

    step_sizes = getattr(move_cfg, "step_sizes", None)
    if step_sizes is None:
        raise ValueError("step_sizes must be provided in move_encs config.")
    else:
        step_sizes = [float(step) for step in step_sizes]

    for step_size in step_sizes:  
        log.info("Decoding encodings for step size=%s", step_size)
        
        step_suffix = _format_float(step_size) if step_size is not None else "0"
        path_out = cache_dir / "moved_encs" / str(config.dir_model.name) / f"alpha{alpha}" / f"step_size{step_suffix}"
        cfg_dataset = config.decode.dataset
        cfg_dataset.path_encodings = str(path_out)
        dataset = instantiate(cfg_dataset).get_subset_by_idxs(decode_idxs)

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=config.experiment.batch_size, 
            shuffle=False
        )

        for batch in tqdm(dataloader, desc=f"Decoding step size {step_size}"):

            batch_img, batch_enc, _, batch_idx = batch
            batch_img = batch_img.to(torch.float32).to(device)
            batch_enc= batch_enc.to(torch.float32).to(device)
                
            with torch.no_grad():

                batch_enc_orig = model.encode(batch_img)
                batch_x_T = model.encode_stochastic(batch_img, batch_enc_orig)
                batch_dec = model.decode(batch_x_T, batch_enc)

            save_batch(
                output_root=output_root,
                image_format=image_format,
                alpha=alpha,
                step_size=step_size,
                batch_indices=batch_idx,
                originals=batch_img,
                decodings=batch_dec,
            )
            
