import torch
import torch.hub
from torchvision.models import vit_b_16, vit_b_32

from utils.lrp_canonizers import VITCanonizer


def get_vit_b_16(
    ckpt_path=None, pretrained=True, n_class: int = None
) -> torch.nn.Module:
    return get_vit(vit_b_16, ckpt_path, pretrained, n_class)


def get_vit_b_32(
    ckpt_path=None, pretrained=True, n_class: int = None
) -> torch.nn.Module:
    return get_vit(vit_b_32, ckpt_path, pretrained, n_class)


def get_vit(
    model_fn, ckpt_path=None, pretrained=True, n_class: int = None
) -> torch.nn.Module:
    if pretrained:
        weights = "IMAGENET1K_V1"
    else:
        weights = None

    model = model_fn(weights=weights)

    if n_class and n_class != 1000:
        model.heads.head = torch.nn.Linear(model.heads.head.in_features, n_class)
    replace_conv = True
    if replace_conv:
        orig_conv = model.conv_proj
        model.conv_proj = torch.nn.Sequential(
            torch.nn.Conv2d(3, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(3, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                3,
                orig_conv.out_channels,
                kernel_size=orig_conv.kernel_size,
                stride=orig_conv.stride,
            ),
        )
    if ckpt_path:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]
        if "module" in list(checkpoint.keys())[0]:
            checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
        model.load_state_dict(checkpoint)

    model.inspection_layer = torch.nn.Identity()
    model.forward = custom_forward.__get__(model)

    return model


def custom_forward(self, x):
    # Reshape and permute the input tensor
    x = self._process_input(x)
    n = x.shape[0]

    # Expand the class token to the full batch
    batch_class_token = self.class_token.expand(n, -1, -1)
    x = torch.cat([batch_class_token, x], dim=1)

    x = self.encoder(x)

    # Classifier "token" as used by standard language architectures
    x = x[:, 0]

    x = self.inspection_layer(x)  # added identity

    x = self.heads(x)

    return x


def get_vit_canonizer():
    return [VITCanonizer()]


if __name__ == "__main__":
    model = get_vit_b_16(pretrained=True)
    print(model)

    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    print(output.shape)
