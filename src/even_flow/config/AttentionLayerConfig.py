from __future__ import annotations
from dataclasses import dataclass

@dataclass
class PatchAttentionLayerConfig:
    """
    Configuration for a patch-based attention layer. This layer divides the input into patches and applies multi-head self-attention to each patch. No activation function is applied after the attention operation.

    Attributes:
        dim: int - the dimension of the input (1, 2, or 3)
        channels: int - the number of input channels
        num_heads: int - the number of attention heads
        patch_size: int - the size of the patches for attention
        norm: str - the normalization method to use (e.g. "group", "batch", or None)
        dropout: float - the dropout rate to apply after the attention operation (training only)
    """
    dim: int # dimension of the input (1, 2, or 3)
    channels: int # number of input channels
    num_heads: int = 4 # number of attention heads
    patch_size: int = 1 # size of the patches for attention
    norm: str = "group" # normalization method
    dropout: float = 0.0 # dropout rate