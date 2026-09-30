# Add to even_flow/config (next to PatchAttentionLayerConfig).
# If your other layer configs share a base class, inherit from it instead of using a bare dataclass.
from dataclasses import dataclass
from typing import Optional


@dataclass
class DiTLayerConfig:
    dim: int                                # spatial dims: 1, 2 or 3
    channels: int                           # channels in = channels out (shape-preserving layer)
    hidden_size: Optional[int] = None       # transformer width; None -> channels
    depth: int = 4                          # number of DiT blocks
    num_heads: int = 4                      # hidden_size must be divisible by this
    patch_size: int = 2                     # every spatial size must be divisible by this
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    time_conditioned: bool = True           # False for models without time (plain UNet, VAE)
    cond_dim: Optional[int] = None          # None -> taken from model_config.time_encoding.time_encoding_dim
    zero_init: bool = True                  # adaLN-Zero: layer is exactly the identity at init

    def convert_to_decoder_layer(self):
        # shape-preserving and symmetric, so the decoder copy is identical
        return self