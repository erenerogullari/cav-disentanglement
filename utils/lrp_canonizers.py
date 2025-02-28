import torch
from torchvision.models.efficientnet import MBConv, FusedMBConv
from torchvision.models.resnet import Bottleneck as ResNetBottleneck, BasicBlock as ResNetBasicBlock
from torchvision.models.vision_transformer import EncoderBlock, Encoder
from torchvision.ops.misc import SqueezeExcitation
from zennit import canonizers as canonizers
from zennit import layer as zlayer
from zennit.canonizers import CompositeCanonizer, SequentialMergeBatchNorm, AttributeCanonizer
from zennit.layer import Sum
from timm.layers.squeeze_excite import SEModule
from timm.layers import BatchNormAct2d
from timm.models.resnet import Bottleneck as ResNetBottleneckTimm, BasicBlock as ResNetBasicBlockTimm
from timm.models.rexnet import LinearBottleneck as RexNetLinearBottleneckTimm

class SignalOnlyGate(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x1, x2):
        return x1 * x2

    @staticmethod
    def backward(ctx, grad_output):
        return torch.zeros_like(grad_output), grad_output


class SECanonizer(canonizers.AttributeCanonizer):
    def __init__(self):
        super().__init__(self._attribute_map)

    @classmethod
    def _attribute_map(cls, name, module):
        if isinstance(module, SqueezeExcitation):
            attributes = {
                'forward': cls.forward.__get__(module),
                'fn_gate': SignalOnlyGate()
            }
            return attributes
        return None
    
    @classmethod
    def _attribute_map(cls, name, module):
        if isinstance(module, SEModule):
            attributes = {
                'forward': cls.forward_timm.__get__(module),
                'fn_gate': SignalOnlyGate()
            }
            return attributes
        return None

    @staticmethod
    def forward_timm(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        if self.add_maxpool:
            # experimental codepath, may remove or change
            x_se = 0.5 * x_se + 0.5 * x.amax((2, 3), keepdim=True)
        x_se = self.fc1(x_se)
        x_se = self.act(self.bn(x_se))
        x_se = self.fc2(x_se)
        scale = self.gate(x_se)
        return self.fn_gate.apply(scale, x)
    
    @staticmethod
    def forward(self, input):
        scale = self._scale(input)
        return self.fn_gate.apply(scale, input)


class MBConvCanonizer(canonizers.AttributeCanonizer):
    '''Canonizer specifically for MBConvBlock of Mobile Net v2 type models.'''

    def __init__(self):
        super().__init__(self._attribute_map)

    @classmethod
    def _attribute_map(cls, name, module):
        if isinstance(module, MBConv):
            attributes = {
                'forward': cls.forward.__get__(module),
                'canonizer_sum': zlayer.Sum()
            }
            return attributes
        
        if isinstance(module, FusedMBConv):
            attributes = {
                'forward': cls.forward_fused.__get__(module),
                'canonizer_sum': zlayer.Sum()
            }
            return attributes
        return None

    @staticmethod
    def forward(self, input):
        result = self.block(input)
        if self.use_res_connect:
            result = self.stochastic_depth(result)

            # result += input
            result = torch.stack([input, result], dim=-1)
            result = self.canonizer_sum(result)
        return result
    
    @staticmethod
    def forward_fused(self, input):
        result = self.block(input)
        if self.use_res_connect:
            result = self.stochastic_depth(result)

            # result += input
            result = torch.stack([input, result], dim=-1)
            result = self.canonizer_sum(result)
        return result




class NewAttention(torch.nn.MultiheadAttention):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, inp):
        result, _ = super().forward(inp, inp, inp, need_weights=False)
        return result


class EncoderBlockCanonizer(canonizers.AttributeCanonizer):
    '''Canonizer specifically for MBConvBlock of Mobile Net v2 type models.'''

    def __init__(self):
        super().__init__(self._attribute_map)

    @classmethod
    def _attribute_map(cls, name, module):
        if isinstance(module, EncoderBlock):

            new_attention = NewAttention(module.self_attention.embed_dim,
                                         module.self_attention.num_heads,
                                         module.self_attention.dropout,
                                         batch_first=True)
            for name, param in module.self_attention.named_parameters():
                if "." in name:
                    getattr(new_attention, name.split(".")[0]).register_parameter(name.split(".")[1], param)
                else:
                    new_attention.register_parameter(name, param)
            attributes = {
                'forward': cls.forward.__get__(module),
                'new_attention': new_attention,
                'sum': zlayer.Sum(),
            }
            return attributes
        return None

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        x = self.new_attention(x)
        x = self.dropout(x)
        x = self.sum(torch.stack([x, input], dim=-1))

        y = self.ln_2(x)
        y = self.mlp(y)
        return self.sum(torch.stack([x, y], dim=-1))


class EncoderCanonizer(canonizers.AttributeCanonizer):
    '''Canonizer specifically for MBConvBlock of Mobile Net v2 type models.'''

    def __init__(self):
        super().__init__(self._attribute_map)

    @classmethod
    def _attribute_map(cls, name, module):
        if isinstance(module, Encoder):
            attributes = {
                'forward': cls.forward.__get__(module),
                'sum': zlayer.Sum(),
            }
            return attributes
        return None

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        input = self.sum(torch.stack([input, self.pos_embedding.expand_as(input)], dim=-1))
        return self.ln(self.layers(self.dropout(input)))


class VITCanonizer(canonizers.CompositeCanonizer):
    def __init__(self):
        super().__init__((
            canonizers.SequentialMergeBatchNorm(),
            EncoderCanonizer(),
            EncoderBlockCanonizer(),
        ))


class ResNetBottleneckCanonizer(AttributeCanonizer):
    '''Canonizer specifically for Bottlenecks of torchvision.models.resnet* type models.'''

    def __init__(self):
        super().__init__(self._attribute_map)

    @classmethod
    def _attribute_map(cls, name, module):
        '''Create a forward function and a Sum module to overload as new attributes for module.

        Parameters
        ----------
        name : string
            Name by which the module is identified.
        module : obj:`torch.nn.Module`
            Instance of a module. If this is a Bottleneck layer, the appropriate attributes to overload are returned.

        Returns
        -------
        None or dict
            None if `module` is not an instance of Bottleneck, otherwise the appropriate attributes to overload onto
            the module instance.
        '''
        if isinstance(module, ResNetBottleneck):
            attributes = {
                'forward': cls.forward.__get__(module),
                'canonizer_sum': Sum(),
            }
            return attributes
        if isinstance(module, ResNetBottleneckTimm):
            attributes = {
                'forward': cls.forward_timm.__get__(module),
                'canonizer_sum': Sum(),
            }
            return attributes
        if isinstance(module, RexNetLinearBottleneckTimm):
            attributes = {
                'forward': cls.forward_timm_rexnet.__get__(module),
                'canonizer_sum': Sum(),
            }
            return attributes
        return None

    @staticmethod
    def forward(self, x):
        '''Modified Bottleneck forward for ResNet.'''
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = torch.stack([identity, out], dim=-1)
        out = self.canonizer_sum(out)

        out = self.relu(out)

        return out
    
    @staticmethod
    def forward_timm(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.drop_block(x)
        x = self.act2(x)
        x = self.aa(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.se is not None:
            x = self.se(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        
        # x += shortcut
        x = torch.stack([shortcut, x], dim=-1)
        x = self.canonizer_sum(x)
        x = self.act3(x)

        return x
    
    @staticmethod
    def forward_timm_rexnet(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        if self.conv_exp is not None:
            x = self.conv_exp(x)
        x = self.conv_dw(x)
        if self.se is not None:
            x = self.se(x)
        x = self.act_dw(x)
        x = self.conv_pwl(x)
        if self.use_shortcut:
            if self.drop_path is not None:
                x = self.drop_path(x)
            x_stacked = torch.stack([shortcut, x[:, 0:self.in_channels]], dim=-1)
            x = torch.cat([self.canonizer_sum(x_stacked), x[:, self.in_channels:]], dim=1)
            # x = torch.cat([x[:, 0:self.in_channels] + shortcut, x[:, self.in_channels:]], dim=1)
        return x


class ResNetBasicBlockCanonizer(AttributeCanonizer):
    '''Canonizer specifically for BasicBlocks of torchvision.models.resnet* type models.'''

    def __init__(self):
        super().__init__(self._attribute_map)

    @classmethod
    def _attribute_map(cls, name, module):
        '''Create a forward function and a Sum module to overload as new attributes for module.

        Parameters
        ----------
        name : string
            Name by which the module is identified.
        module : obj:`torch.nn.Module`
            Instance of a module. If this is a BasicBlock layer, the appropriate attributes to overload are returned.

        Returns
        -------
        None or dict
            None if `module` is not an instance of BasicBlock, otherwise the appropriate attributes to overload onto
            the module instance.
        '''
        if isinstance(module, ResNetBasicBlock):
            attributes = {
                'forward': cls.forward.__get__(module),
                'canonizer_sum': Sum(),
            }
            return attributes
        
        if isinstance(module, ResNetBasicBlockTimm):
            attributes = {
                'forward': cls.forward_timm.__get__(module),
                'canonizer_sum': Sum(),
            }
            return attributes
        return None

    @staticmethod
    def forward_timm(self, x):
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.drop_block(x)
        x = self.act1(x)
        x = self.aa(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.se is not None:
            x = self.se(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        
        x = torch.stack([shortcut, x], dim=-1)
        x = self.canonizer_sum(x)
        # x += shortcut
        x = self.act2(x)
        return x
    
    @staticmethod
    def forward(self, x):
        '''Modified BasicBlock forward for ResNet.'''
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = torch.stack([identity, out], dim=-1)
        out = self.canonizer_sum(out)

        if hasattr(self, 'last_conv'):
            out = self.last_conv(out)
            out = out + 0

        out = self.relu(out)

        return out

class BatchNormActCanonizer(AttributeCanonizer):
    '''Canonizer specifically for SE layers.'''

    def __init__(self):
        super().__init__(self._attribute_map)

    @classmethod
    def _attribute_map(cls, name, module):
        if isinstance(module, BatchNormAct2d):
            new_bn = torch.nn.BatchNorm2d(module.num_features).eval()
            new_bn.bias.data = module.bias.data
            new_bn.weight.data = module.weight.data
            new_bn.running_mean = module.running_mean
            new_bn.running_var = module.running_var
            attributes = {
                'forward': cls.forward.__get__(module),
                'bn': new_bn
            }
            return attributes
        return None

    @staticmethod
    def forward(self, x):
        '''Modified SE forward for SENetworks.'''
        x = self.bn(x)
        x = self.drop(x)
        x = self.act(x)
        return x
    
class ResNetCanonizer(CompositeCanonizer):
    '''Canonizer for torchvision.models.resnet* type models. This applies SequentialMergeBatchNorm, as well as
    add a Sum module to the Bottleneck modules and overload their forward method to use the Sum module instead of
    simply adding two tensors, such that forward and backward hooks may be applied.'''

    def __init__(self):
        super().__init__((
            SequentialMergeBatchNorm(),
            ResNetBottleneckCanonizer(),
            ResNetBasicBlockCanonizer(),
        ))

class EfficientNetBNCanonizer(canonizers.CompositeCanonizer):
    def __init__(self):
        super().__init__((
            SECanonizer(),
            MBConvCanonizer(),
            canonizers.SequentialMergeBatchNorm()
        ))

class RexNetCanonizer(canonizers.CompositeCanonizer):
    def __init__(self):
        super().__init__((
            SECanonizer(),
            ResNetBottleneckCanonizer(),
            BatchNormActCanonizer(),
            SequentialMergeBatchNorm(),
        ))