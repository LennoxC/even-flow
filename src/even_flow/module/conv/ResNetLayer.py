import torch
from even_flow.module.conv.ConvLayer import ConvLayer, UpsampleConvLayer, DownsampleConvLayer

class ResNetLayer(torch.nn.Module):
    """
    A res-net inspired block. Uses two convolutional layers with a skip connection.
    - The number of channels may change, and if so a 1x1 convolution is used to match the number of channels for the skip connection.
    - No skip connection is used if upsampling or downsampling is used.
    - The activation function is applied after the first convolutional layer.
    - Additional kwargs are passed to the ConvLayer.
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
                    **kwargs):
        super().__init__()
        self.dim = dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.activation = activation
        self.norm = norm
        self.sampling = sampling

        if sampling == "upsample":
            self.conv1 = UpsampleConvLayer(dim=dim, in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, norm=norm, **kwargs)
            self.conv2 = ConvLayer(dim=dim, in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, norm=norm, **kwargs)
            self.skip_conv = None # no skip connection for upsampling
        elif sampling == "downsample":
            self.conv1 = DownsampleConvLayer(dim=dim, in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, norm=norm, **kwargs)
            self.conv2 = ConvLayer(dim=dim, in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, norm=norm, **kwargs)
            self.skip_conv = None # no skip connection for downsampling
        else:
            self.conv1 = ConvLayer(dim=dim, in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, norm=norm, **kwargs)
            self.conv2 = ConvLayer(dim=dim, in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, norm=norm, **kwargs)
            if in_channels != out_channels:
                self.skip_conv = ConvLayer(dim=dim, in_channels=in_channels, out_channels=out_channels, kernel_size=1, **kwargs) # 1x1 convolution to match channels for skip connection
            else:
                self.skip_conv = None

        self.activation = getattr(torch.nn, activation)()

    def forward(self, x):
        skip_x = x
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        if self.skip_conv is not None:
            skip_x = self.skip_conv(skip_x)
            return x + skip_x
        else:
            return x
