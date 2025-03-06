import torch
import torch.nn as nn
import copy


# Define a custom module to perform the flattening and max pooling
class FlattenAndMaxPool(nn.Module):
    def forward(self, x):
        x = x.flatten(start_dim=2)   # Flatten to shape [-1, C, H, W] -> [-1, C, H*W]
        x, _ = torch.max(x, dim=2)   # Max along the last dimension, shape [-1, C]
        return x
    
def truncate_and_extend(model_name, model: nn.Module, layer_name: str, pooling_module: nn.Module, W: torch.Tensor) -> nn.Module:
    if model_name == 'vgg16':
        return truncate_and_extend_vgg16(model, layer_name, pooling_module, W)
    elif model_name == 'resnet18':
        truncate_and_extend_resnet18(model, pooling_module, W)
    else:
        raise NotImplementedError()

# Helper fn to truncate and extend the model to have CAVs on the last layer
def truncate_and_extend_vgg16(model: nn.Module, layer_name: str, pooling_module: nn.Module, W: torch.Tensor) -> nn.Module :
    # Deep copy of the model
    model = copy.deepcopy(model)

    module_name, layer_idx = layer_name.split(".")
    layer_idx = int(layer_idx)
    
    # Truncate the module's layers up to the specified index
    target_module = getattr(model, module_name)
    truncated_module = nn.Sequential(*list(target_module.children())[:layer_idx + 1])
    setattr(model, module_name, truncated_module)

    # Remove any subsequent layers
    submodules = list(model._modules.keys())
    target_module_index = submodules.index(module_name)
    for module in submodules[target_module_index + 1:]:
        if module != 'input_identity':  # Ensure 'input_identity' is not removed
            delattr(model, module)

    # Infer the number of channels from the last convolutional layer
    last_conv_layer = list(truncated_module.children())[-1]
    num_channels = last_conv_layer.out_channels

    # Check the shape of W for compatibility with the number of channels
    num_features = W.shape[1]
    if num_features != num_channels:
        raise ValueError(f"Weight matrix W must have {num_channels} columns, but has {num_features}.")

    # Extend the model with the given pooling layer and the weights matrix
    model.add_module('custom_pool', pooling_module)
    linear_layer = nn.Linear(num_channels, W.shape[0], bias=False)
    linear_layer.weight = nn.Parameter(W.to(dtype=torch.float32))
    model.add_module('custom_linear', linear_layer)

    # Finally implement a new forward method
    def forward_modified(self, x):
        x = self.input_identity(x)  
        x = self.features(x)
        x = self.custom_pool(x)
        x = self.custom_linear(x)  
        return x
    
    # And attach to the modified model
    model.forward = forward_modified.__get__(model, nn.Module)
    
    return model

def truncate_and_extend_resnet18(model: nn.Module, pooling_module: nn.Module, W: torch.Tensor) -> nn.Module:
    # Deep copy the model
    model = copy.deepcopy(model)

    # Verify compatibility of W with the output channel
    num_channels = 512
    if W.shape[1] != num_channels:
        raise ValueError(f"Weight matrix W must have {num_channels} columns, but has {W.shape[1]}.")
    
    # Remove layers beyond the truncation point
    for layer_name in ["last_relu", "avgpool", "fc"]:
        if hasattr(model, layer_name):
            delattr(model, layer_name)

    # Extend the model with the given pooling layer and the weights matrix
    model.add_module('custom_pool', pooling_module)
    linear_layer = nn.Linear(num_channels, W.shape[0], bias=False)
    linear_layer.weight = nn.Parameter(W.to(dtype=torch.float32))
    model.add_module('custom_linear', linear_layer)

    # Finally implement a new forward method
    def forward(self, x):
        x = self.input_identity(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.identity_0(x)  # added identity
        x = self.relu_0(x)

        x = self.layer2(x)
        x = self.identity_1(x)  # added identity
        x = self.relu_1(x)

        x = self.layer3(x)
        x = self.identity_2(x)  # added identity
        x = self.relu_2(x)

        x = self.layer4(x)
        x = self.last_conv(x)

        # Truncated part
        # x = self.last_relu(x)  # added identity
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)

        x = self.custom_pool(x)
        x = self.custom_linear(x)
        return x
    
    # And attach to the modified model
    model.forward = forward.__get__(model)

    return model