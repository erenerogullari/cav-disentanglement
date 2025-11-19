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

    for img_idx, orig_img, dec_img in zip(index_list, originals, decodings):
        img_dir = alpha_dir / f"img{int(img_idx)}"
        img_dir.mkdir(parents=True, exist_ok=True)

        original_path = img_dir / f"original.{image_format}"
        if not original_path.exists():

            torchvision.utils.save_image(orig_img, original_path)

        step_suffix = _format_float(step_size) if step_size is not None else "0"
        decoded_path = img_dir / f"step_size{step_suffix}.{image_format}"
        torchvision.utils.save_image(dec_img, decoded_path)


def run_decode(config: DictConfig, moved_encodings: Dict[float, Dict[Optional[float], torch.Tensor]], moved_idxs: torch.Tensor) -> None:
    experiment_cfg = config.experiment

    log.info("Seeding RNGs with %s", experiment_cfg.seed)
    seed_everything(int(experiment_cfg.seed))

    device = torch.device(experiment_cfg.device)
    log.info("Using device %s", device)

    move_cfg = config.move_encs
    max_images = getattr(move_cfg, "num_images", None)
    if max_images is not None and moved_idxs.size(0) > int(max_images):
        max_images = int(max_images)
        perm = torch.randperm(moved_idxs.size(0))[:max_images]
        moved_idxs = moved_idxs[perm]
    elif max_images is not None and moved_idxs.size(0) <= int(max_images):
        raise ValueError(f"Not enough samples to move for num_images={max_images}")

    model = build_model(config, device)
    dataset = instantiate(config.dataset).get_subset_by_idxs(moved_idxs.tolist())  # type: ignore

    image_format = getattr(experiment_cfg, "format", "png")
    output_root = Path("results") / "diffae" / "decodings" / config.dir_model.name
    output_root.mkdir(parents=True, exist_ok=True)

    moved_indices = moved_idxs.long().tolist()
    log.info(
        "Starting decoding for %d samples across %d direction models.",
        len(moved_indices),
        len(moved_encodings),
    )

    results = {}
    for alpha, step_dict in sorted(moved_encodings.items(), key=lambda item: item[0]):
        log.info("Decoding encodings for alpha=%s", alpha)

        for step_size, encs in sorted(step_dict.items(), key=lambda item: item[0]): # type: ignore
            log.info("Decoding encodings for step size=%s", step_size)
            encs = encs[moved_idxs].to(device)
            dataloader = DataLoader(
                TensorDataset(
                    torch.stack([sample for sample, _ in dataset]),
                    encs,
                    moved_idxs
                ), 
                batch_size=experiment_cfg.batch_size, 
                shuffle=False
            )

            with torch.no_grad():
                for batch_imgs, batch_encs, idxs in tqdm(dataloader, desc=f"Decoding step size {step_size}"):
                    batch_encs = batch_encs.to(device=device, dtype=torch.float32)
                    batch_imgs = batch_imgs.to(device=device, dtype=torch.float32)

                    batch_encs_orig = model.encode(batch_imgs)
                    batch_x_T = model.encode_stochastic(batch_imgs, batch_encs_orig)
                    batch_dec = model.decode(batch_x_T, batch_encs)

                    save_batch(
                        output_root=output_root,
                        image_format=image_format,
                        alpha=alpha,
                        step_size=step_size,
                        batch_indices=idxs,
                        originals=dataset.reverse_normalization(batch_imgs).float() / 255.0,
                        decodings=batch_dec,
                    )
            
