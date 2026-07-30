from __future__ import annotations
from dataclasses import dataclass

@dataclass
class ProbabilisticLayerConfig:
    """
    This is the layer which maps the encoder output to the probabilistic latent space.
    Attributes:
        latent_dim: int - the in_channels of the latent space. This is multiplied by 2 to get the number of channels in the latent space, since the latent space is parameterized by a mean and a log variance. This should match the out_channels of the last encoder layer
        channels: int - the number of channels in the latent space
        dim: int - the dimension of the input (1, 2, or 3)
        logvar_clamp: tuple[float, float] - the range to clamp the log variance to. This is used to prevent numerical instability when sampling from the latent space. The log variance is clamped to this range before exponentiating to get the variance. The default is (-30.0, 20.0).
    """
    latent_dim: int
    channels: int
    dim: int = 2
    logvar_clamp: tuple[float, float] = (-30.0, 20.0)