import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union
import os

class LeNet5(nn.Module):
    def __init__(self, num_classes=2, in_channels: int = 1, input_size: Union[int, Tuple[int, int]] = 32):
        super(LeNet5, self).__init__()
        self.in_channels = int(in_channels)
        if isinstance(input_size, (tuple, list)):
            if len(input_size) != 2:
                raise ValueError("input_size iterable must have length 2 (height, width).")
            self.input_size = (int(input_size[0]), int(input_size[1]))
        else:
            size_int = int(input_size)
            self.input_size = (size_int, size_int)

        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        feature_dim = self._infer_feature_dim()

        self.fc1 = nn.Linear(feature_dim, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def _infer_feature_dim(self) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, self.in_channels, self.input_size[0], self.input_size[1])
            out = self.pool1(F.tanh(self.conv1(dummy)))
            out = self.pool2(F.tanh(self.conv2(out)))
            return out.flatten(1).shape[1]

    def forward(self, x):
        x = self.pool1(F.tanh(self.conv1(x)))
        x = self.pool2(F.tanh(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)
        return x
    

def get_lenet5(ckpt_path=None, pretrained=True, n_class: int = 0, in_channels: int = 1, input_size: int = 32) -> torch.nn.Module:

    if n_class == 0:
        raise ValueError("n_class must be specified for LeNet5 model.")
    
    model = LeNet5(
        num_classes=n_class if n_class else 2,
        in_channels=in_channels,
        input_size=input_size,
    )

    if pretrained and ckpt_path:
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


def get_lenet_canonizer():
    return []
