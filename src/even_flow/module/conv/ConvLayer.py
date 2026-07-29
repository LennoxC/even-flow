import torch
from abc import ABC, abstractmethod

# changes for dataclasses (WIP):
# - norm is now a string instead of a boolean. Must be passed into the Conv layer.
# - new variable: separable: bool = False
# - activation is no longer passed into ConvBase

class ConvBase(torch.nn.Module):
    """
    A base convolutional layer (1d, 2d, 3d).
    - This is an abstract class and should not be instantiated directly. Use ConvLayer, ConvUpsampleLayer, or ConvDownsampleLayer instead.
    - The convolutional layer can be separable or not. If separable, the convolution is implemented as a depthwise convolution followed by a pointwise convolution.
    - The convolutional layer can be followed by a normalization layer (group or batch normalization) if specified.
    - Activations are not included in this base class, and should be added as a separate layer if desired. ResNetBlocks implement an activation, and are composed of two ConvBase layers.
    """

    def __init__(self,
                 dim: int, 
                 in_channels: int, 
                 out_channels: int, 
                 kernel_size: int = 3, 
                 norm: str = "group",
                 separable: bool = False,
                 **kwargs):
        super().__init__()
        self.dim = dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.separable = separable

        # Set padding to maintain the same spatial dimensions after convolution by default. This can be overridden by specifying a different padding in kwargs.
        if not hasattr(self, 'padding'):
            self.padding = kernel_size // 2

        if self.separable:
            self.conv = torch.nn.Sequential(
                getattr(torch.nn, f"Conv{dim}d")(in_channels, in_channels, kernel_size, padding=self.padding, groups=in_channels, **kwargs),
                getattr(torch.nn, f"Conv{dim}d")(in_channels, out_channels, kernel_size=1, **kwargs)
            )
        else:
            self.conv = getattr(torch.nn, f"Conv{dim}d")(in_channels, out_channels, kernel_size, padding=self.padding, **kwargs)

        self.norm = self._norm(norm, out_channels, dim)

    def forward(self, x):
        x = self.preprocess(x)
        x = self.conv(x)
        x = self.normalize(x)
        x = self.postprocess(x)
        return x

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

class ConvUpsampleLayer(ConvBase):
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
            if self.separable: # if separable, then self.conv is a sequential of two convolutions.
                self.conv[0] = getattr(torch.nn, f"ConvTranspose{self.dim}d")(self.in_channels, self.in_channels, kernel_size=self.kernel_size, stride=self.upsample_factor, padding=self.padding, groups=self.in_channels)
                self.conv[1] = getattr(torch.nn, f"ConvTranspose{self.dim}d")(self.in_channels, self.out_channels, kernel_size=1, stride=1)
            else:
                self.conv = getattr(torch.nn, f"ConvTranspose{self.dim}d")(self.in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.upsample_factor, padding=self.padding)
        else:    
            self.upsample = torch.nn.Upsample(scale_factor=self.upsample_factor, mode=upsample_method)

    def __str__(self):
        return f"ConvUpsampleLayer{self.dim}d, in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.conv.kernel_size}, upsample_method={self.upsample_method})"

    def preprocess(self, x):
        x = self.upsample(x)
        return x

class ConvDownsampleLayer(ConvBase):
    """
    A basic convolutional layer (1d, 2d, 3d) with an activation function and downsampling.
    Include downsampling using a specified method (e.g., max pooling, average pooling, strided).
    """
    def __init__(self, 
                    downsample_method: str = "strided",
                    sample_factor: int = 2,
                    **kwargs):
        super().__init__(**kwargs)

        self.downsample_method = downsample_method
        self.downsample_factor = sample_factor
        if downsample_method == "strided":
            if self.separable: # if separable, then self.conv is a sequential of two convolutions.
                self.conv[0].stride = self.downsample_factor
                self.conv[1].stride = 1
            else:
                self.conv.stride = self.downsample_factor
        else:
            self.downsample = getattr(
                torch.nn, 
                f"MaxPool{self.dim}d")(kernel_size=self.downsample_factor) if downsample_method == "max" else getattr(torch.nn, f"AvgPool{self.dim}d"
            )(kernel_size=self.downsample_factor)

    def __str__(self):
        return f"ConvDownsampleLayer{self.dim}d, in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.conv.kernel_size}, downsample_method={self.downsample_method})"

    def postprocess(self, x):
        if self.downsample_method != "strided":
            x = self.downsample(x)
        
        # if self.downsample_method == "strided", downsampling is already handled in the convolutional layer, so no need to downsample again.
        return x