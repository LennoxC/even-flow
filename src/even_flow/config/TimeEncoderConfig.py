from __future__ import annotations
from dataclasses import dataclass

@dataclass(kw_only=True)
class TimeEncoderConfig:
    """
    Configuration for a time encoder. This includes a sinusoidal time encoding and a linear layer for projecting time to the latent space.

    Args:
        time_encoding_dim: int - the dimensionality of the sinusoidal time encoding
        channel_dim: int - the dimensionality of the channel space after projecting the time encoding
        activation: str - the activation function to use after the linear projection of the time encoding
    """
    encoding_strategy: str = "Sinusoidal" # the strategy for encoding time (currently only supports "sinusoidal")
    time_encoding_dim: int = 64
    activation: str = "GELU"
