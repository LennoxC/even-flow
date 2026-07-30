import math
import torch
import torch.nn.functional as F

class PatchAttentionLayer(torch.nn.Module):
    """
    A patch-based multi-head self-attention layer (1d, 2d, 3d), intended to be inserted
    between convolutional stages of an encoder/decoder (e.g. at the bottleneck of a VAE,
    or at low-resolution stages where attention is affordable).

    - Pre-norm: GroupNorm is applied before attention
    - Residual: output is x + attention(norm(x)) so the block can be skipped early in training.
    - Zero-initialized output projection: at init, this layer is the identity function,
      which avoids destabilizing the surrounding conv stack when attention is first introduced.
    - The spatial/volumetric dims are split into non-overlapping patches
      (like ViT) before computing attention, so the token count is (N / patch_size^dim)
      rather than N. Set patch_size=1 for standard pixel-wise self-attention.

    Only supports self-attention over a single feature map (no cross-attention / conditioning).
    Input and output shapes are identical: (B, C, *spatial_dims).
    """

    def __init__(self,
                 dim: int,
                 channels: int,
                 num_heads: int = 4,
                 patch_size: int = 1,
                 norm: str = "group",
                 dropout: float = 0.0,
                 **kwargs):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels ({channels}) must be divisible by num_heads ({num_heads})")

        self.dim = dim
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.patch_size = patch_size
        self.dropout = dropout

        self.norm = self._norm(norm, channels, dim)

        # patch embedding: gathers a patch_size^dim neighborhood into the channel dim via a strided conv,
        # then projects back out. If patch_size == 1, this degenerates to pointwise (1x1) convs.
        conv_cls = getattr(torch.nn, f"Conv{dim}d")
        self.to_qkv = conv_cls(channels, channels * 3, kernel_size=patch_size, stride=patch_size, padding=0)

        # output projection: zero-initialized so this block starts as an identity function
        self.to_out = conv_cls(channels, channels, kernel_size=1)
        torch.nn.init.zeros_(self.to_out.weight)
        torch.nn.init.zeros_(self.to_out.bias)

        if patch_size > 1:
            # projects attended patch tokens back up to full spatial resolution
            conv_transpose_cls = getattr(torch.nn, f"ConvTranspose{dim}d")
            self.unpatch = conv_transpose_cls(channels, channels, kernel_size=patch_size, stride=patch_size, padding=0)
        else:
            self.unpatch = None

    def _norm(self, norm, channels, dim):
        if norm == "group":
            return torch.nn.GroupNorm(min(32, channels), channels)
        if norm == "batch":
            return getattr(torch.nn, f"BatchNorm{dim}d")(channels)
        else:
            raise ValueError(f"Invalid normalization type: {norm}. Supported types are 'group' and 'batch'.")

    def __str__(self):
        return (f"PatchAttentionLayer{self.dim}d, channels={self.channels}, "
                f"num_heads={self.num_heads}, patch_size={self.patch_size})")

    def forward(self, x):
        residual = x
        x = self.norm(x)

        spatial_shape = x.shape[2:]
        if any(s % self.patch_size != 0 for s in spatial_shape):
            raise ValueError(
                f"Spatial dims {tuple(spatial_shape)} must be divisible by patch_size ({self.patch_size})"
            )

        qkv = self.to_qkv(x)  # (B, 3*C, *patched_spatial_dims)
        patched_shape = qkv.shape[2:]
        b = qkv.shape[0]

        # flatten spatial/volumetric dims into a single token dimension: (B, 3*C, N)
        qkv = qkv.flatten(2)
        q, k, v = qkv.chunk(3, dim=1)  # each (B, C, N)

        # split channels into heads: (B, num_heads, head_dim, N) -> (B, num_heads, N, head_dim)
        def to_heads(t):
            return t.view(b, self.num_heads, self.head_dim, -1).transpose(-1, -2)

        q, k, v = to_heads(q), to_heads(k), to_heads(v)

        attn = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
        )  # (B, num_heads, N, head_dim)

        # merge heads back: (B, num_heads, N, head_dim) -> (B, C, N)
        attn = attn.transpose(-1, -2).reshape(b, self.channels, -1)
        attn = attn.view(b, self.channels, *patched_shape)

        if self.unpatch is not None:
            attn = self.unpatch(attn)

        out = self.to_out(attn)
        return residual + out