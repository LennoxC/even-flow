from dataclasses import dataclass

@dataclass(kw_only=True)
class AutoencoderConfig:
    input_dim: tuple[int, int, int] # input dims [channels, height, width]
    output_dim: tuple[int, int, int] = None # output dims [channels, height, width]. If None, output dims will be the same as input dims

    activation: str = "GeLU" # activation function

@dataclass
class ConvolutionalAutoencoderConfig(AutoencoderConfig):
    encoder_layers: list[DownsampleLayerConfig] # list of layer configurations
    decoder_layers: list[UpsampleLayerConfig] = None # list of layer configurations. If None, decoder will be the reverse of the encoder with nearest upsampling

    group_norm: bool = False # whether to use group normalization
    batch_norm: bool = False # whether to use batch normalization

@dataclass
class LayerConfig:
    dim: int # dimension of the convolution (1, 2, or 3)
    in_channels: int # number of input channels
    out_channels: int # number of output channels
    kernel_size: int # size of the convolution kernel
    activation: str = "GeLU" # activation function. Overrides the default activation function in the AutoencoderConfig if specified

@dataclass
class DownsampleLayerConfig(LayerConfig):
    downsample_method: str = "max" # downsampling method (max, avg, or strided)

@dataclass
class UpsampleLayerConfig(LayerConfig):
    upsample_method: str = "nearest" # upsampling method (nearest, bilinear, trilinear, or transposed)