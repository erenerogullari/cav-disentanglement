import sys
import torch
from torch import nn
from models.src_diffae.renderer import render_condition
from models.src_diffae.templates import *


def make_config(config_name):
    if config_name == 'ffhq256_autoenc':
        return ffhq256_autoenc()
    elif config_name == 'ffhq128_autoenc_130m':
        return ffhq128_autoenc_130M()
    else:
        raise NotImplementedError('Invalid config name.')

class DiffAE(nn.Module):

    def __init__(self, config_name: str, path_ckpt: str, T_encode: int, T_decode: int, **kwargs) -> None:
        super().__init__()
        self.config = make_config(config_name)
        self.model = self.config.make_model_conf().make_model()
        self.sampler_encode = self.config._make_diffusion_conf(T_encode).make_sampler()
        self.sampler_decode = self.config._make_diffusion_conf(T_decode).make_sampler()
        self.load_ckpt(path_ckpt)


    def load_ckpt(self, path):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
        ckpt_ema = {'.'.join(k.split('.')[1:]): v for k, v in ckpt.items() if "ema" in k}
        self.model.load_state_dict(ckpt_ema)
        self.requires_grad_(False)
        self.model.eval()


    def encode(self, x):
        if x.min() >= 0. and x.max() <= 1.:
            x = (x - 0.5) * 2
        return self.model.encoder.forward(x)


    def encode_stochastic(self, x, cond):        
        if x.min() >= 0. and x.max() <= 1.:
            x = (x - 0.5) * 2
        out = self.sampler_encode.ddim_reverse_sample_loop(
            self.model, x, model_kwargs = {'cond': cond})
        return out['sample']

    
    def decode(self, noise, cond):
        sampler = self.sampler_decode
        pred_img = render_condition(self.config,
                                    self.model,
                                    noise,
                                    sampler = sampler,
                                    cond = cond)
        pred_img = (pred_img + 1) / 2
        return pred_img


    def reconstruct(self, x):
        cond = self.encode(x)
        x_T = self.encode_stochastic(x, cond)
        return self.decode(x_T, cond)