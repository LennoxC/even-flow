import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod

# changes for dataclasses (WIP):
# - norm is now a string instead of a boolean. Must be passed into the Conv layer.
# - new variable: separable: bool = False
# - activation is no longer passed into ConvBase

class ConvBase(torch.nn.Module):
    """
    A base convolutional layer (1d, 2d, 3d).
    - This is an abstract class and should not be instantiated directly. Use ConvLayer, UpsampleConvLayer, or DownsampleConvLayer instead.
    - The convolutional layer can be separable or not. If separable, the convolution is implemented as a depthwise convolution followed by a pointwise convolution.
    - The convolutional layer can be followed by a normalization layer (group or batch normalization) if specified.
    - Activations are not included in this base class, and should be added as a separate layer if desired. ResNetLayers implement an activation, and are composed of two ConvBase layers.
    """

    def __init__(self, dim, in_channels, out_channels, kernel_size=3, norm="group",
                 separable=False, receives_skip=False, emit_skip=False,
                 stride: int = 1, pad_mode: str = "reflect", **kwargs):
        super().__init__()
        self.dim = dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.separable = separable
        self.receives_skip = receives_skip
        self.emit_skip = emit_skip
        self.stride = stride
        self.pad_mode = pad_mode
 
        # "same"-style padding: total = k - stride, split asymmetrically if odd.
        total = kernel_size - stride
        left = total // 2
        right = total - left
        self.padding = left  # kept for the transposed-conv path below
        self.pad_tuple = (left, right) * dim   # F.pad wants (W_l, W_r, H_l, H_r, ...)
 
        Conv = getattr(torch.nn, f"Conv{dim}d")
        if separable:
            self.conv = torch.nn.Sequential(
                Conv(in_channels, in_channels, kernel_size, stride=stride, padding=0,
                     groups=in_channels, **kwargs),
                Conv(in_channels, out_channels, kernel_size=1, **kwargs),
            )
        else:
            self.conv = Conv(in_channels, out_channels, kernel_size, stride=stride,
                             padding=0, **kwargs)
 
        self.norm = self._norm(norm, out_channels, dim)

    def forward(self, x):
        x = self.preprocess(x)
        if any(self.pad_tuple):
            x = F.pad(x, self.pad_tuple, mode=self.pad_mode)
        x = self.conv(x)
        x = self.normalize(x)
        return self.postprocess(x)

    def preprocess(self, x):
        return x

    def postprocess(self, x):
        return x

    def normalize(self, x):
        if self.norm is not None:
            x = self.norm(x)
        return x

    def _norm(self, norm, channels, dim):
        if norm == "group":
            return torch.nn.GroupNorm(1, channels)
        if norm == "batch":
            return getattr(torch.nn, f"BatchNorm{dim}d")(channels)
        else:
            raise ValueError(f"Invalid normalization type: {norm}. Supported types are 'group' and 'batch'.")
        return None

class ConvLayer(ConvBase):
    """
    A basic convolutional layer (1d, 2d, 3d) with an activation function.
    As this layer does not include upsampling or downsampling, a skip connection can be used.
    The number of channels may change, and if so a 1x1 convolution is used to match the number of channels for the skip connection.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.in_channels != self.out_channels:
            self.skip_conv = getattr(torch.nn, f"Conv{self.dim}d")(self.in_channels, self.out_channels, kernel_size=1)
        else:
            self.skip_conv = None

    # override the forward method to include a skip connection
    def forward(self, x):
        skip_x = x
        x = super().forward(x)
        if self.skip_conv is not None:
            skip_x = self.skip_conv(skip_x)
        return x + skip_x

class UpsampleConvLayer(ConvBase):
    """
    A basic convolutional layer (1d, 2d, 3d) with upsampling.
    Include upsampling using a specified method (e.g., nearest, bilinear, trilinear) or transposed convolution.
    """
    def __init__(self, 
                    upsample_method: str = "nearest",
                    sample_factor: int = 2,
                    **kwargs):
        super().__init__(**kwargs)

        # dimensionality checks for upsampling methods
        self.upsample_method = upsample_method
        if self.upsample_method == "bilinear" and self.dim != 2:
            raise ValueError(f"bilinear upsampling is only supported for 2D convolutions, but got dim={self.dim}")
        if self.upsample_method == "trilinear" and self.dim != 3:
            raise ValueError(f"trilinear upsampling is only supported for 3D convolutions, but got dim={self.dim}")

        self.upsample_factor = sample_factor
        if upsample_method == "transposed":
            self.pad_tuple = (0,) * (2 * self.dim)   # transposed conv handles its own padding
            pad = (self.kernel_size - self.upsample_factor) // 2
            if self.separable: # if separable, then self.conv is a sequential of two convolutions.
                self.conv[0] = getattr(torch.nn, f"ConvTranspose{self.dim}d")(self.in_channels, self.in_channels, kernel_size=self.kernel_size, stride=self.upsample_factor, padding=pad, groups=self.in_channels)
                self.conv[1] = getattr(torch.nn, f"ConvTranspose{self.dim}d")(self.in_channels, self.out_channels, kernel_size=1, stride=1)
            else:
                self.conv = getattr(torch.nn, f"ConvTranspose{self.dim}d")(self.in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.upsample_factor, padding=pad)
        else:    
            self.upsample = torch.nn.Upsample(scale_factor=self.upsample_factor, mode=upsample_method)

    def __str__(self):
        return f"UpsampleConvLayer{self.dim}d, in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.conv.kernel_size}, upsample_method={self.upsample_method})"

    def preprocess(self, x):
        x = self.upsample(x)
        return x

class DownsampleConvLayer(ConvBase):
    """
    A basic convolutional layer (1d, 2d, 3d) with an activation function and downsampling.
    Include downsampling using a specified method (e.g., max pooling, average pooling, strided).
    """
    def __init__(self, downsample_method="strided", sample_factor=2, **kwargs):
        stride = sample_factor if downsample_method == "strided" else 1
        super().__init__(stride=stride, **kwargs)
        self.downsample_method = downsample_method
        self.downsample_factor = sample_factor
        if downsample_method != "strided":
            self.downsample = (torch.nn.MaxPool if downsample_method == "max" else torch.nn.AvgPool)
            self.downsample = getattr(torch.nn, f"{'Max' if downsample_method=='max' else 'Avg'}Pool{self.dim}d")(kernel_size=sample_factor)

    def __str__(self):
        return f"DownsampleConvLayer{self.dim}d, in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.conv.kernel_size}, downsample_method={self.downsample_method})"

    def postprocess(self, x):
        if self.downsample_method != "strided":
            x = self.downsample(x)
        
        # if self.downsample_method == "strided", downsampling is already handled in the convolutional layer, so no need to downsample again.
        return x