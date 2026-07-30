from __future__ import annotations
from dataclasses import dataclass
import warnings

@dataclass
class ConvLayerConfig:
    """
    Configuration for a convolutional layer. This can be used to configure both encoder and decoder layers, as well as static convolutional layers.
    The intention of the ConvLayerConfig is to be a generic configuration for a convolutional layer, which can be used in both encoder and decoder networks.
    This will not upsample/downsample the input. Consider using DownsampleConvLayerConfig or UpsampleConvLayerConfig for layers that downsample or upsample the input, respectively.

    Attributes:
        dim: int - the dimension of the convolution (1, 2, or 3)
        in_channels: int - the number of input channels
        out_channels: int - the number of output channels
        kernel_size: int - the size of the convolution kernel
        norm: str - the normalization method to use (e.g. "group", "batch", or None)
        separable: bool - whether to use separable convolutions. This is intended for use in 3D convolutions: if separable is True and dim < 3, a warning will be emitted.
        receives_skip: bool - whether this layer receives a skip connection from the encoder. This is used in the decoder to determine which layers should receive skip connections from the encoder.
    """
    dim: int # dimension of the convolution (1, 2, or 3)
    in_channels: int # number of input channels
    out_channels: int # number of output channels
    kernel_size: int # size of the convolution kernel
    norm: str = "group" # normalization method
    separable: bool = False # whether to use separable convolutions
    receives_skip: bool = False # whether to receive a skip connection from the encoder

    def __post_init__(self):
        if self.separable and self.dim < 3:
            warnings.warn(
                f"ConvLayerConfig: separable is True but dim ({self.dim}) < 3. "
                f"Separable convolutions are intended for use in 3D convolutions."
            )

    def convert_to_decoder_layer(self):
        """
        Utility method to flip the in_channels and out_channels. This is useful for converting an encoder layer config to a decoder layer config.
        """
        return ConvLayerConfig(
            dim=self.dim,
            in_channels=self.out_channels,
            out_channels=self.in_channels,
            kernel_size=self.kernel_size,
            norm=self.norm,
            separable=self.separable
        )

@dataclass
class DownsampleConvLayerConfig(ConvLayerConfig):
    """
    Inherits from ConvLayerConfig.
    This is a configuration for a convolutional layer that downsamples the input. This can be used in encoder networks to reduce the spatial dimensions of the input.

    Attributes:
        downsample_method: str - the method to use for downsampling (e.g. "max", "avg", or "strided"). If "strided", the convolution will be applied with a stride equal to the sample_factor. If "max" or "avg", a max pooling or average pooling layer will be applied after the convolution.
        sample_factor: int - the factor by which to downsample the input. This is used to determine the stride of the convolution or the kernel size of the pooling layer, depending on the downsample_method.
        emit_skip: bool - whether this layer emits a skip connection to the decoder. This is used in the encoder to determine which layers should emit skip connections to the decoder. If True, the output of this layer will be passed to the decoder as a skip connection.
        + the attributes from ConvLayerConfig are inherited: dim, in_channels, out_channels, kernel_size, norm, separable, and receives_skip.
    """
    downsample_method: str = "max" # downsampling method (max, avg, or strided)
    sample_factor: int = 2 # downsampling factor
    emit_skip: bool = False # whether to emit a skip connection from this layer to the decoder

    # utility method to convert to an UpsampleConvLayerConfig, for use when the decoder is the reverse of the encoder
    def convert_to_decoder_layer(self):
        return UpsampleConvLayerConfig(
            dim=self.dim,
            in_channels=self.out_channels,
            out_channels=self.in_channels,
            kernel_size=self.kernel_size,
            norm=self.norm,
            separable=self.separable,
            upsample_method="nearest" if self.downsample_method == "max" else "bilinear",
            sample_factor=self.sample_factor
        )


@dataclass
class UpsampleConvLayerConfig(ConvLayerConfig):
    """
    Inherits from ConvLayerConfig.
    This is a configuration for a convolutional layer that upsamples the input. This can be used in decoder networks to increase the spatial dimensions of the input.
    Attributes:
        upsample_method: str - the method to use for upsampling (e.g. "nearest", "bilinear", "trilinear", or "transposed"). If "transposed", a transposed convolution will be applied. If "nearest", "bilinear", or "trilinear", an upsampling layer will be applied before the convolution.
        sample_factor: int - the factor by which to upsample the input. This is used to determine the scale factor of the upsampling layer or the stride of the transposed convolution, depending on the upsample_method.
        receives_skip: bool - whether this layer receives a skip connection from the encoder. This is used in the decoder to determine which layers should receive skip connections from the encoder. If True, the input to this layer will be concatenated with the corresponding skip connection from the encoder.
        + the attributes from ConvLayerConfig are inherited: dim, in_channels, out_channels, kernel_size, norm, separable, and receives_skip.
    """
    upsample_method: str = "nearest" # upsampling method (nearest, bilinear, trilinear, or transposed)
    sample_factor: int = 2 # upsampling factor
    receives_skip: bool = False # whether to receive a skip connection from the encoder

    def __post_init__(self):
        # raise an error if receives skip is true, and the inherited emit_skip is also true, since this would be a conflict
        if self.receives_skip and getattr(self, 'emit_skip', False):
            raise ValueError(
                f"UpsampleConvLayerConfig: receives_skip is True but emit_skip is also True. "
                f"These attributes are mutually exclusive."
            )

