import torch
import torch.hub
from torchvision.models import vgg16, vgg16_bn, vgg11, vgg13, vgg13_bn, vgg11_bn
from torchvision.models.detection import SSD300_VGG16_Weights, ssd300_vgg16
from zennit.torchvision import VGGCanonizer


def get_vgg16(ckpt_path=None, pretrained=True, n_class: int = None) -> torch.nn.Module:
    return get_vgg(vgg16, ckpt_path, pretrained, n_class)


class VGG16SSDCOCO(torch.nn.Module):
    """Tensor-output wrapper around the COCO-trained SSD VGG feature stack."""

    def __init__(self, features: torch.nn.Sequential) -> None:
        super().__init__()
        self.input_identity = torch.nn.Identity()
        self.features = features
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_identity(x)
        x = self.features(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)


def _unwrap_checkpoint(checkpoint):
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    if not isinstance(checkpoint, dict):
        raise ValueError("Expected an SSD state-dict checkpoint.")
    if checkpoint and next(iter(checkpoint)).startswith("module."):
        checkpoint = {
            key.removeprefix("module."): value for key, value in checkpoint.items()
        }
    return checkpoint


def get_vgg16_ssd_coco(
    ckpt_path=None,
    pretrained=True,
    n_class: int | None = None,
) -> torch.nn.Module:
    """Load the SSD300 COCO VGG backbone through ``conv5_3``.

    ``n_class`` is accepted for compatibility with the shared model-loader API;
    the returned model is a feature extractor rather than a classifier.
    """
    del n_class
    if ckpt_path is not None:
        detector = ssd300_vgg16(weights=None, weights_backbone=None)
        checkpoint = torch.load(ckpt_path, weights_only=True, map_location="cpu")
        detector.load_state_dict(_unwrap_checkpoint(checkpoint))
    else:
        weights = SSD300_VGG16_Weights.COCO_V1 if pretrained else None
        detector = ssd300_vgg16(
            weights=weights,
            weights_backbone=None,
        )

    # SSD splits VGG after conv4_3. Rejoin the original convolutional blocks
    # through conv5_3/ReLU so inspection names match torchvision VGG16.
    feature_layers = list(detector.backbone.features.children())
    conv5_layers = list(detector.backbone.extra[0].children())[:7]
    model = VGG16SSDCOCO(torch.nn.Sequential(*feature_layers, *conv5_layers))
    return model.eval()


def get_vgg16_bn(ckpt_path=None, pretrained=True, n_class=None) -> torch.nn.Module:
    return get_vgg(vgg16_bn, ckpt_path, pretrained, n_class)


def get_vgg13(ckpt_path=None, pretrained=True, n_class: int = None) -> torch.nn.Module:
    return get_vgg(vgg13, ckpt_path, pretrained, n_class)


def get_vgg13_bn(ckpt_path=None, pretrained=True, n_class: int = None) -> torch.nn.Module:
    return get_vgg(vgg13_bn, ckpt_path, pretrained, n_class)


def get_vgg11(ckpt_path=None, pretrained=True, n_class: int = None) -> torch.nn.Module:
    return get_vgg(vgg11, ckpt_path, pretrained, n_class)


def get_vgg11_bn(ckpt_path=None, pretrained=True, n_class: int = None) -> torch.nn.Module:
    return get_vgg(vgg11_bn, ckpt_path, pretrained, n_class)


def get_vgg(model_fn, ckpt_path=None, pretrained=True, n_class: int = None) -> torch.nn.Module:
    if pretrained:
        weights = "IMAGENET1K_V1"
    else:
        weights = None

    model = model_fn(weights=weights)

    if n_class and n_class != 1000:
        model.classifier[-1] = torch.nn.Linear(4096, n_class, bias=True)
    if ckpt_path:
        checkpoint = torch.load(ckpt_path, weights_only=True, map_location="cpu")
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]
        if "module" in list(checkpoint.keys())[0]:
            checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
        checkpoint = {k.replace("classifier.last", "classifier.6"): v for k, v in checkpoint.items()}  # ISIC MODEL
        model.load_state_dict(checkpoint)
    model.input_identity = torch.nn.Identity()
    model.forward = forward_modified.__get__(model)
    return model


def get_vgg_canonizer():
    return [VGGCanonizer()]

def forward_modified(self, x: torch.Tensor) -> torch.Tensor:
    x = self.input_identity(x)
    x = self.features(x)
    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    x = self.classifier(x)
    return x
