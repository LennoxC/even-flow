import torch
from abc import ABC, abstractmethod

class ConvBase(torch.nn.Module):
    def __init__(self,
                 dim: int, 
                 in_channels: int, 
                 out_channels: int, 
                 activation: str,
                 kernel_size: int = 3, 
                 **kwargs):
        super().__init__()
        self.dim = dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.activation = activation

        self.conv = getattr(torch.nn, f"Conv{dim}d")(in_channels, out_channels, kernel_size, **kwargs)
        self.activation = getattr(torch.nn, activation)()

    def forward(self, x):
        x = self.preprocess(x)
        x = self.conv(x)
        x = self.postprocess(x)
        x = self.activation(x)
        return x

    def preprocess(self, x):
        return x

    def postprocess(self, x):
        return x

class ConvLayer(ConvBase):
    """
    A basic convolutional layer (1d, 2d, 3d) with an activation function.
    Include upsampling and downsampling options using pooling or transposed convolution.
    """
    pass

class ConvUpsampleLayer(ConvBase):
    """
    A basic convolutional layer (1d, 2d, 3d) with an activation function and upsampling.
    Include upsampling using a specified method (e.g., nearest, bilinear, trilinear) or transposed convolution.
    """
    def __init__(self, 
                    upsample_method: str = "nearest",
                    **kwargs):
        super().__init__(**kwargs)

        self.upsample_method = upsample_method
        self.upsample = torch.nn.Upsample(scale_factor=2, mode=upsample_method)

    def __str__(self):
        return f"ConvUpsampleLayer{self.dim}d, in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.conv.kernel_size}, activation={self.activation}, upsample_method={self.upsample_method})"

    def postprocess(self, x):
        x = self.upsample(x)
        return x

class ConvDownsampleLayer(ConvBase):
    """
    A basic convolutional layer (1d, 2d, 3d) with an activation function and downsampling.
    Include downsampling using a specified method (e.g., max pooling, average pooling).
    """
    def __init__(self, 
                    downsample_method: str = "max",
                    **kwargs):
        super().__init__(**kwargs)

        self.downsample_method = downsample_method
        self.downsample = getattr(
            torch.nn, 
            f"MaxPool{self.dim}d")(kernel_size=2) if downsample_method == "max" else getattr(torch.nn, f"AvgPool{self.dim}d"
        )(kernel_size=2)

    def __str__(self):
        return f"ConvDownsampleLayer{self.dim}d, in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.conv.kernel_size}, activation={self.activation}, downsample_method={self.downsample_method})"

    def postprocess(self, x):
        x = self.downsample(x)
        return x