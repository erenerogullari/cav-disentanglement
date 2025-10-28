from typing import Union
from models.src_diffae.model.unet import BeatGANsUNetModel, BeatGANsUNetConfig
from models.src_diffae.model.unet_autoenc import BeatGANsAutoencConfig, BeatGANsAutoencModel

Model = Union[BeatGANsUNetModel, BeatGANsAutoencModel]
ModelConfig = Union[BeatGANsUNetConfig, BeatGANsAutoencConfig]
