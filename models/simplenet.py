import torch
import torch.nn as nn
from typing import Tuple, Union
import os

class SimpleNet(nn.Module):
    def __init__(self, num_classes=2, in_channels: int = 3, input_size: Union[int, Tuple[int, int]] = 224):
        super(SimpleNet, self).__init__()
        self.in_channels = int(in_channels)
        if isinstance(input_size, (tuple, list)):
            if len(input_size) != 2:
                raise ValueError("input_size iterable must have length 2 (height, width).")
            self.input_size = (int(input_size[0]), int(input_size[1]))
        else:
            size_int = int(input_size)
            self.input_size = (size_int, size_int)

        layers = []
        channels = [64, 64, 64, 128, 128, 128]
        current_channels = self.in_channels
        for idx, out_channels in enumerate(channels):
            layers.append(nn.Conv2d(current_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            if idx < 3:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            current_channels = out_channels

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        feature_dim = self._infer_feature_dim()
        self.classifier = nn.Linear(feature_dim, num_classes)

    def _infer_feature_dim(self) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, self.in_channels, self.input_size[0], self.input_size[1])
            out = self.features(dummy)
            out = self.avgpool(out)
            return out.flatten(1).shape[1]

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    

def get_simplenet(ckpt_path=None, pretrained=True, n_class: int = 0, in_channels: int = 3, input_size: int = 224) -> torch.nn.Module:

    if n_class == 0:
        raise ValueError("n_class must be specified for SimpleNet model.")
    
    model = SimpleNet(
        num_classes=n_class,
        in_channels=in_channels,
        input_size=input_size,
    )

    if ckpt_path:
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"The model has no pretrained weights available or the checkpoint path name ({ckpt_path}) is not correct.")
        
        checkpoint = torch.load(ckpt_path, weights_only=True)
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]
        if "module" in list(checkpoint.keys())[0]:
            checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
        model.load_state_dict(checkpoint)

    return model


def get_simplenet_canonizer():
    return []


if __name__ == "__main__":
    # Print layer names
    model = get_simplenet(n_class=10, in_channels=3, input_size=32)
    for name, module in model.named_modules():
        print(f"{name}: {module}")
