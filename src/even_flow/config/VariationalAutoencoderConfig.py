from dataclasses import dataclass

# changes for dataclasses (WIP):
# - norm is now a string instead of a boolean. Must be passed into the Conv layer.
# - new variable: separable: bool = False
# - activation is no longer passed into ConvBase

"""
Params required for ResNetBlock dataclass:
dim: int, 
in_channels: int, 
out_channels: int, 
kernel_size: int = 3, 
norm: str = "group",
separable: bool = False,
upsample: bool = False,
downsample: bool = False,
sample_factor: int = 2,
upsample_method: str = "nearest",
downsample_method: str = "strided",
activation: str = "GELU",
"""

@dataclass(kw_only=True)
class VariationalAutoencoderConfig:
    input_dim: tuple[int, int, int] # input dims [channels, height, width]
    output_dim: tuple[int, int, int] = None # output dims [channels, height, width]. If None, output dims will be the same as input dims

    activation: str = "GELU" # activation function

@dataclass
class ConvolutionalVariationalAutoencoderConfig(VariationalAutoencoderConfig):
    encoder_layers: list[DownsampleConvLayerConfig] # list of layer configurations
    decoder_layers: list[UpsampleConvLayerConfig] = None # list of layer configurations. If None, decoder will be the reverse of the encoder with nearest upsampling

    norm: str = "group"

@dataclass
class ResNetLayerConfig:
    dim: int
    in_channels: int
    out_channels: int
    kernel_size: int = 3
    norm: str = None # normalization method (group, batch, or None)
    separable: bool = False
    sampling: str = None # "upsample" or "downsample" or None (same resolution)
    sample_factor: int = 2
    upsample_method: str = "nearest"
    downsample_method: str = "strided"
    activation: str = "GELU"

@dataclass
class ConvLayerConfig:
    dim: int # dimension of the convolution (1, 2, or 3)
    in_channels: int # number of input channels
    out_channels: int # number of output channels
    kernel_size: int # size of the convolution kernel
    norm: str = "group" # normalization method
    separable: bool = False # whether to use separable convolutions

@dataclass
class DownsampleConvLayerConfig(ConvLayerConfig):
    downsample_method: str = "max" # downsampling method (max, avg, or strided)
    sample_factor: int = 2 # downsampling factor

@dataclass
class UpsampleConvLayerConfig(ConvLayerConfig):
    upsample_method: str = "nearest" # upsampling method (nearest, bilinear, trilinear, or transposed)
    sample_factor: int = 2 # upsampling factor