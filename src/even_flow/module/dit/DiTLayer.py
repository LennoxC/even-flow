"""
Diffusion Transformer (DiT) layer for convolutional U-shaped models.

Takes a feature map (B, C, *spatial), patchifies it into tokens, runs `depth` DiT blocks
(adaLN-Zero conditioning on the time embedding), then un-patchifies back to (B, C, *spatial).
The output is added to the input (residual), so the layer is shape-preserving and can be dropped
into the middle of a UNet without changing any channel counts.

Time conditioning follows the same contract as TimeInjection: it consumes the (B, cond_dim)
embedding produced by TimeEmbedding. Layers with `takes_time_emb = True` are called as
`layer(x, emb)` by the UNet runners instead of `layer(x)`.
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

_CONV = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}
_CONV_T = {1: nn.ConvTranspose1d, 2: nn.ConvTranspose2d, 3: nn.ConvTranspose3d}


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    # x: (B, N, D); shift/scale: (B or 1, D)
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def _axis_sincos(n: int, d: int, device) -> torch.Tensor:
    """Fixed 1D sin-cos embedding, (n, d). d must be even."""
    pos = torch.arange(n, device=device, dtype=torch.float32)
    omega = 1.0 / (10000 ** (torch.arange(d // 2, device=device, dtype=torch.float32) / (d // 2)))
    angles = pos[:, None] * omega[None, :]
    return torch.cat([angles.sin(), angles.cos()], dim=1)


def sincos_pos_embed(hidden_size: int, grid, device=None, dtype=torch.float32) -> torch.Tensor:
    """
    Fixed N-D sin-cos positional embedding for a token grid, returned as (prod(grid), hidden_size).
    The channels are split evenly across the spatial axes; any remainder is zero padded, so
    hidden_size does not need to be divisible by 2 * ndim. Being fixed and computed on the fly,
    it works for any grid size (e.g. a different resolution at inference).
    """
    grid = tuple(int(g) for g in grid)
    ndim = len(grid)
    d = 2 * (hidden_size // (2 * ndim))
    if d < 2:
        raise ValueError(f"hidden_size={hidden_size} is too small for a {ndim}D positional embedding.")
    index = torch.meshgrid(*[torch.arange(n, device=device) for n in grid], indexing="ij")
    parts = [_axis_sincos(n, d, device)[idx.reshape(-1)] for n, idx in zip(grid, index)]
    emb = torch.cat(parts, dim=1)
    if emb.shape[1] < hidden_size:
        emb = F.pad(emb, (0, hidden_size - emb.shape[1]))
    return emb.to(dtype)


class _SelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads}).")
        self.num_heads = num_heads
        self.dropout = dropout
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads, D // self.num_heads).permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)
        return self.proj(out.transpose(1, 2).reshape(B, N, D))


class DiTBlock(nn.Module):
    """
    Pre-norm transformer block. With cond_dim set, uses adaLN-Zero: the conditioning embedding
    produces (shift, scale, gate) for both the attention and MLP branches. Without cond_dim it
    is a standard pre-norm block with learnable LayerNorm affine parameters.
    """
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, dropout: float, cond_dim: Optional[int]):
        super().__init__()
        affine = cond_dim is None  # modulation replaces the affine when conditioned
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=affine, eps=1e-6)
        self.attn = _SelfAttention(hidden_size, num_heads, dropout)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=affine, eps=1e-6)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_size),
            nn.Dropout(dropout),
        )
        self.adaLN = (
            nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 6 * hidden_size)) if cond_dim is not None else None
        )

    def forward(self, x: torch.Tensor, emb: Optional[torch.Tensor]) -> torch.Tensor:
        if self.adaLN is None:
            x = x + self.attn(self.norm1(x))
            return x + self.mlp(self.norm2(x))
        shift1, scale1, gate1, shift2, scale2, gate2 = self.adaLN(emb).chunk(6, dim=-1)
        x = x + gate1.unsqueeze(1) * self.attn(_modulate(self.norm1(x), shift1, scale1))
        x = x + gate2.unsqueeze(1) * self.mlp(_modulate(self.norm2(x), shift2, scale2))
        return x


class DiTLayer(nn.Module):
    """
    Args:
        dim: number of spatial dims (1, 2 or 3).
        channels: channels of the incoming/outgoing feature map.
        hidden_size: transformer width. Defaults to `channels`.
        depth: number of DiT blocks.
        num_heads: attention heads (hidden_size must be divisible by it).
        patch_size: patch edge length. Every spatial size must be divisible by it.
        mlp_ratio: MLP expansion ratio.
        dropout: dropout in attention and MLP.
        cond_dim: size of the conditioning (time) embedding. None -> unconditioned layer
            (usable in models without time, e.g. a VAE).
        zero_init: zero-initialise the adaLN modulation and the output projection (adaLN-Zero),
            so the layer is exactly the identity at initialisation.
    """
    takes_time_emb = True  # UNet runners call layer(x, emb) for layers with this flag

    def __init__(
        self,
        dim: int,
        channels: int,
        hidden_size: Optional[int] = None,
        depth: int = 4,
        num_heads: int = 4,
        patch_size: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        cond_dim: Optional[int] = None,
        zero_init: bool = True,
    ):
        super().__init__()
        if dim not in _CONV:
            raise ValueError(f"dim must be 1, 2 or 3, got {dim}.")
        hidden_size = hidden_size or channels

        # NOTE: deliberately `channels`, not `in_channels`: FlowMatchingUNet adds a TimeInjection to
        # every decoder layer that has `in_channels`, and this layer does its own time conditioning.
        self.dim = dim
        self.channels = channels
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.cond_dim = cond_dim
        self.requires_emb = cond_dim is not None

        self.patch_embed = _CONV[dim](channels, hidden_size, kernel_size=patch_size, stride=patch_size)
        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_size, num_heads, mlp_ratio, dropout, cond_dim) for _ in range(depth)]
        )
        self.final_norm = nn.LayerNorm(hidden_size, elementwise_affine=cond_dim is None, eps=1e-6)
        self.final_adaLN = (
            nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 2 * hidden_size)) if cond_dim is not None else None
        )
        self.unpatch = _CONV_T[dim](hidden_size, channels, kernel_size=patch_size, stride=patch_size)

        self._init_weights(zero_init)

    def _init_weights(self, zero_init: bool):
        def basic(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.apply(basic)
        w = self.patch_embed.weight
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        nn.init.zeros_(self.patch_embed.bias)

        if zero_init:
            for block in self.blocks:
                if block.adaLN is not None:
                    nn.init.zeros_(block.adaLN[-1].weight)
                    nn.init.zeros_(block.adaLN[-1].bias)
            if self.final_adaLN is not None:
                nn.init.zeros_(self.final_adaLN[-1].weight)
                nn.init.zeros_(self.final_adaLN[-1].bias)
            nn.init.zeros_(self.unpatch.weight)
            nn.init.zeros_(self.unpatch.bias)

    def forward(self, x: torch.Tensor, emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.dim() != self.dim + 2:
            raise ValueError(f"Expected a {self.dim + 2}D input (B, C, *spatial) for dim={self.dim}, got shape {tuple(x.shape)}.")
        if x.size(1) != self.channels:
            raise ValueError(f"Channel mismatch: x has {x.size(1)} channels, DiTLayer expects {self.channels}.")
        if any(s % self.patch_size for s in x.shape[2:]):
            raise ValueError(f"Spatial size {tuple(x.shape[2:])} is not divisible by patch_size={self.patch_size}.")
        if self.requires_emb:
            if emb is None:
                raise ValueError("This DiTLayer is time-conditioned (cond_dim is set) but no embedding was passed.")
            if emb.size(-1) != self.cond_dim:
                raise ValueError(f"Embedding has {emb.size(-1)} features, DiTLayer expects cond_dim={self.cond_dim}.")

        B = x.size(0)
        tokens = self.patch_embed(x)  # (B, D, *grid)
        grid = tokens.shape[2:]
        tokens = tokens.flatten(2).transpose(1, 2)  # (B, N, D)
        tokens = tokens + sincos_pos_embed(self.hidden_size, grid, tokens.device, tokens.dtype)

        for block in self.blocks:
            tokens = block(tokens, emb)

        if self.final_adaLN is not None:
            shift, scale = self.final_adaLN(emb).chunk(2, dim=-1)
            tokens = _modulate(self.final_norm(tokens), shift, scale)
        else:
            tokens = self.final_norm(tokens)

        tokens = tokens.transpose(1, 2).reshape(B, self.hidden_size, *grid)
        return x + self.unpatch(tokens)