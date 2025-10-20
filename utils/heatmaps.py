import numpy as np
import torch

def normalize_heatmap(heatmap:torch.Tensor, channel_avg: bool = False) -> torch.Tensor:
    """Normalize heatmap by averaging over channels and normalizing with max absolute value
    Args:
        heatmap (torch.Tensor): Input heatmap of shape (C, H, W).
        channel_avg (bool): Whether to average over channels. Default is False.
    Returns:
        torch.Tensor: Normalized heatmap.
    """
    if channel_avg:
        heatmap = heatmap.mean(dim=0)  # Shape (H, W)

    abs_max_heatmap = torch.abs(heatmap).max() + 1e-10  # Avoid division by zero
    heatmap = heatmap / abs_max_heatmap

    return heatmap

def conormalize_heatmaps(heatmaps: torch.Tensor, channel_avg: bool = False) -> torch.Tensor:
    """Conormalize heatmaps by a global max-abs value.

    Args:
        heatmaps: Tensor shaped (N, C, H, W) or (N, H, W).
        channel_avg: If True, average over the channel axis before normalizing.

    Returns:
        torch.Tensor: Normalized heatmaps with the same device/dtype.
    """
    if not isinstance(heatmaps, torch.Tensor):
        raise TypeError("`heatmaps` must be a torch.Tensor.")

    data = heatmaps.detach()
    squeeze_channel = False

    if data.dim() == 3:
        data = data.unsqueeze(1)
        squeeze_channel = True
    elif data.dim() != 4:
        raise ValueError("`heatmaps` must have 3 or 4 dimensions.")

    if channel_avg:
        data = data.mean(dim=1, keepdim=False)
    else:
        squeeze_channel = False  # preserve (N, C, H, W)

    abs_max = data.abs().amax()
    if abs_max <= 0:
        abs_max = abs_max + 1e-10

    normalized = data / abs_max

    if not channel_avg and squeeze_channel:
        normalized = normalized.unsqueeze(1)

    return normalized


def _coerce_to_tensor(data) -> torch.Tensor:
    """Convert nested sequences, numpy arrays, or tensors into a torch.Tensor."""
    if isinstance(data, torch.Tensor):
        return data.detach()
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    if isinstance(data, (list, tuple)):
        if len(data) == 0:
            raise ValueError("`heatmaps` cannot be empty.")
        tensors = [_coerce_to_tensor(item) for item in data]
        base = tensors[0]
        tensors = [t.to(device=base.device, dtype=base.dtype) for t in tensors]
        try:
            return torch.stack(tensors)
        except RuntimeError as exc:
            raise ValueError("Heatmaps must share the same shape to be stacked.") from exc
    return torch.as_tensor(data)


def _normalize_batch(batch: torch.Tensor, channel_avg: bool, conormalize: bool) -> torch.Tensor:
    """Normalize a batch of heatmaps either individually or jointly."""
    if batch.dim() < 2:
        raise ValueError("Heatmaps must have at least 2 dimensions (H, W).")

    if conormalize:
        if batch.shape[0] == 1:
            normalized = normalize_heatmap(batch[0], channel_avg=channel_avg).unsqueeze(0)
        else:
            normalized = conormalize_heatmaps(batch, channel_avg=channel_avg)
    else:
        normalized = torch.stack([
            normalize_heatmap(hm, channel_avg=channel_avg) for hm in batch
        ])

    return normalized


def _heatmap_to_display_array(hm: torch.Tensor, channel_avg: bool) -> np.ndarray:
    """Convert a single heatmap tensor into a 2D numpy array for plotting."""
    if isinstance(hm, torch.Tensor):
        tensor = hm.detach()
    else:
        tensor = torch.as_tensor(hm)

    if tensor.dim() == 4 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)

    if tensor.dim() == 3:
        if tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        elif channel_avg:
            tensor = tensor.mean(dim=0)
        else:
            raise ValueError("Heatmap has multiple channels; set `channel_avg=True` to average them.")

    if tensor.dim() != 2:
        raise ValueError("Heatmap must be 2D after optional channel reduction.")

    return tensor.cpu().numpy()