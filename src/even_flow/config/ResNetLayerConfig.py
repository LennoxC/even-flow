from __future__ import annotations
from dataclasses import dataclass

@dataclass
class ResNetLayerConfig:
    """
    This is a ResNet-inspired convolutional block. This can be used as an upsampling or downsampling block in the encoder or decoder, or as a static convolutional block.

    A forward pass would get transformed as follows:
    - Convolution Layer
    - Normalization Layer (if norm is not None)
    - Activation Layer
    - Convolution Layer
    - Normalization Layer (if norm is not None)
    - Add the input to the output of the second normalization layer (residual connection)

    Attributes:
        dim: int - the dimension of the convolution (1, 2, or 3)
        in_channels: int - the number of input channels
        out_channels: int - the number of output channels
        kernel_size: int - the size of the convolution kernel
        norm: str - the normalization method to use (e.g. "group", "batch", or None)
        separable: bool - whether to use separable convolutions. This is intended for use in 3D convolutions: if separable is True and dim < 3, a warning will be emitted.
        sampling: str - the sampling method to use (e.g. "upsample", "downsample", or None). If "upsample", the block will upsample the input. If "downsample", the block will downsample the input. If None, the block will not change the spatial dimensions of the input.
        sample_factor: int - the factor by which to upsample or downsample the input. This is used to determine the scale factor of the upsampling layer or the stride of the downsampling layer, depending on the sampling method.
        upsample_method: str - the method to use for upsampling (e.g. "nearest", "bilinear", "trilinear", or "transposed"). If "transposed", a transposed convolution will be applied. If "nearest", "bilinear", or "trilinear", an upsampling layer will be applied before the convolution.
        downsample_method: str - the method to use for downsampling (e.g. "max", "avg", or "strided"). If "strided", the convolution will be applied with a stride equal to the sample_factor. If "max" or "avg", a max pooling or average pooling layer will be applied after the convolution.
        activation: str - the activation function to use. This can match the class name of any pytorch activation function (e.g. "ReLU", "GELU", "LeakyReLU", etc.). The activation function will be instantiated with default parameters.
        emit_skip: bool - whether this layer emits a skip connection to the decoder. This is used in the encoder to determine which layers should emit skip connections to the decoder.
        receives_skip: bool - whether this layer receives a skip connection from the encoder. This is used in the decoder to determine which layers should receive skip connections from the encoder.
    """
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
    emit_skip: bool = False # whether to emit a skip connection from this layer to the decoder
    receives_skip: bool = False # whether to receive a skip connection from the encoder

    def convert_to_decoder_layer(self):
        """
        Utility method to convert an encoder layer config to a decoder layer config.
        This is useful for when the decoder is the reverse of the encoder.
        """
        if self.sampling == "downsample":
            return ResNetLayerConfig(
                dim=self.dim,
                in_channels=self.out_channels,
                out_channels=self.in_channels,
                kernel_size=self.kernel_size,
                norm=self.norm,
                separable=self.separable,
                sampling="upsample",
                sample_factor=self.sample_factor,
                upsample_method="nearest" if self.downsample_method == "max" else "bilinear",
                activation=self.activation,
                receives_skip=self.emit_skip,
                emit_skip=self.receives_skip
            )
        elif self.sampling == "upsample":
            return ResNetLayerConfig(
                dim=self.dim,
                in_channels=self.out_channels,
                out_channels=self.in_channels,
                kernel_size=self.kernel_size,
                norm=self.norm,
                separable=self.separable,
                sampling="downsample",
                sample_factor=self.sample_factor,
                downsample_method="max" if self.upsample_method == "nearest" else "avg",
                activation=self.activation,
                receives_skip=self.emit_skip,
                emit_skip=self.receives_skip
            )
        else:
            return ResNetLayerConfig(
                dim=self.dim,
                in_channels=self.out_channels,
                out_channels=self.in_channels,
                kernel_size=self.kernel_size,
                norm=self.norm,
                separable=self.separable,
                sampling=None,
                activation=self.activation,
                receives_skip=self.emit_skip,
                emit_skip=self.receives_skip
            )