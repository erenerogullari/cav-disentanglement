import copy
import numpy as np
import torch
from experiments.utils.activations import get_features
import torchvision.transforms as T
from skimage.filters import threshold_otsu
from models import requires_lxt_localization
from utils.reslrp_torchvision import attribute_concept


def get_localizations(
    x,
    cav,
    attribution,
    composite,
    config,
    device,
    *,
    model_name="",
    model=None,
):
    if requires_lxt_localization(model_name):
        if model is None:
            raise ValueError("ViT localization requires the classification model.")
        hms = attribute_concept(
            model,
            x.to(device),
            cav,
            layer_name=config["layer_name"],
        )
        return None, hms.detach().cpu().clamp(min=0)

    _config = copy.deepcopy(config)
    _config["cav_mode"] = "cavs_full"
    _config["device"] = device
    act = get_features(x.to(device), _config, attribution).detach()
    init_rel = (act.clamp(min=0) * cav[..., None, None].to(device)).to(device)
    attr = attribution(x.to(device).requires_grad_(), [{}], composite, start_layer=config["layer_name"], init_rel=init_rel)
    hms = attr.heatmap.detach().cpu().clamp(min=0)
    return attr, hms

def binarize_heatmaps(hms, kernel_size=7, sigma=8.0, thresholding="otsu", percentile=92):
    gaussian = T.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
    heatmaps_binary = []
    for hm in hms:
        hm_smooth = gaussian(hm.clamp(min=0)[None])[0].numpy()
        if thresholding == "otsu":
            thresh = threshold_otsu(hm_smooth)
        else:
            thresh = np.percentile(hm_smooth, percentile)
        heatmaps_binary.append((hm_smooth > thresh).astype(np.uint8))
    heatmaps_binary = np.array(heatmaps_binary)
    return torch.Tensor(heatmaps_binary).type(torch.uint8)
