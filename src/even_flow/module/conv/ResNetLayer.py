from typing import Optional

import torch
from even_flow.module.conv.ConvLayer import ConvLayer, UpsampleConvLayer, DownsampleConvLayer


class ResNetLayer(torch.nn.Module):
    """
    A res-net inspired block. Uses two convolutional layers with a skip connection.
    - The number of channels may change, and if so a 1x1 convolution is used to match the number of channels for the skip connection.
    - No skip connection is used if upsampling or downsampling is used.
    - The activation function is applied after the first convolutional layer.
    - Additional kwargs are passed to the ConvLayer.

    NEW (opt-in):
    - cond_dim: if set, the block applies FiLM (scale, shift) from the time embedding after conv1,
      zero-initialised so it starts as a no-op. The layer then has `takes_time_emb = True`, so the
      UNet runners call layer(x, emb), including in the ENCODER (which previously never saw time),
      and FlowMatchingUNet skips its additive TimeInjection for this layer.
      With cond_dim=None there are no extra parameters and the state_dict is unchanged.

    Full-resolution stems / refinement need no flag: use sampling=None (same resolution) together
    with the existing emit_skip (encoder) / receives_skip (decoder) handling in the UNet runners.
    """
    def __init__(self,
                    dim: int,
                    in_channels: int,
                    out_channels: int,
                    kernel_size: int = 3,
                    norm: str = "group",
                    separable: bool = False,
                    sampling: str = None, # "upsample" or "downsample" or None (same resolution)
                    sample_factor: int = 2,
                    upsample_method: str = "nearest",
                    downsample_method: str = "strided",
                    activation: str = "GELU",
                    receives_skip: bool = False,
                    emit_skip: bool = False,
                    cond_dim: Optional[int] = None,
                    **kwargs):
        super().__init__()
        self.dim = dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.activation = activation
        self.norm = norm
        self.sampling = sampling
        self.receives_skip = receives_skip
        self.emit_skip = emit_skip
        self.cond_dim = cond_dim
        self.takes_time_emb = cond_dim is not None

        if sampling == "upsample":
            self.conv1 = UpsampleConvLayer(dim=dim, in_channels=in_channels, out_channels=out_channels,
                                            kernel_size=kernel_size, norm=norm, separable=separable,
                                            upsample_method=upsample_method, sample_factor=sample_factor, **kwargs)
            self.conv2 = ConvLayer(dim=dim, in_channels=out_channels, out_channels=out_channels,
                                    kernel_size=kernel_size, norm=norm, separable=separable, **kwargs)
            self.skip_conv = None
        elif sampling == "downsample":
            self.conv1 = DownsampleConvLayer(dim=dim, in_channels=in_channels, out_channels=out_channels,
                                                kernel_size=kernel_size, norm=norm, separable=separable,
                                                downsample_method=downsample_method, sample_factor=sample_factor, **kwargs)
            self.conv2 = ConvLayer(dim=dim, in_channels=out_channels, out_channels=out_channels,
                                    kernel_size=kernel_size, norm=norm, separable=separable, **kwargs)
            self.skip_conv = None
        else:
            self.conv1 = ConvLayer(dim=dim, in_channels=in_channels, out_channels=out_channels,
                                    kernel_size=kernel_size, norm=norm, separable=separable, **kwargs)
            self.conv2 = ConvLayer(dim=dim, in_channels=out_channels, out_channels=out_channels,
                                    kernel_size=kernel_size, norm=norm, separable=separable, **kwargs)
            self.skip_conv = (ConvLayer(dim=dim, in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=1, **kwargs) if in_channels != out_channels else None)

        self.activation = getattr(torch.nn, activation)()

        # optional time FiLM. Plain attribute (None) when disabled -> no state_dict entries.
        self.film = None
        if cond_dim is not None:
            self.film = torch.nn.Sequential(torch.nn.SiLU(), torch.nn.Linear(cond_dim, 2 * out_channels))
            torch.nn.init.zeros_(self.film[-1].weight)
            torch.nn.init.zeros_(self.film[-1].bias)

    def forward(self, x, emb: Optional[torch.Tensor] = None):
        skip_x = x
        x = self.conv1(x)
        if self.film is not None:
            if emb is None:
                raise ValueError("ResNetLayer was built with cond_dim but no time embedding was passed.")
            scale, shift = self.film(emb).chunk(2, dim=-1)           # (B or 1, C) each
            pad = (None,) * self.dim
            x = x * (1 + scale[(slice(None), slice(None)) + pad]) + shift[(slice(None), slice(None)) + pad]
        x = self.activation(x)
        x = self.conv2(x)
        if self.skip_conv is not None:
            skip_x = self.skip_conv(skip_x)
            return x + skip_x
        else:
            return x