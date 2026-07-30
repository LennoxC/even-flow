from dataclasses import dataclass

# TODO: Group these into different files for clarity

@dataclass(kw_only=True)
class VariationalAutoencoderConfig:
    input_dim: tuple[int, int, int] # input dims [channels, height, width]
    output_dim: tuple[int, int, int] = None # output dims [channels, height, width]. If None, output dims will be the same as input dims

    activation: str = "GELU" # activation function

@dataclass
class ConvolutionalVariationalAutoencoderConfig(VariationalAutoencoderConfig):
    # TODO: These classes are misleading as they could be a range of different layers types.
    encoder_layers: list[DownsampleConvLayerConfig] # list of layer configurations
    probabilistic_layer: ProbabilisticLayerConfig # configuration for the probabilistic layer (latent moments)
    decoder_layers: list[UpsampleConvLayerConfig] = None # list of layer configurations. If None, decoder will be the reverse of the encoder with nearest upsampling
    static_dim: tuple[int, int, int] = None # input dims for the static encoder [channels, height, width]. If None, no static encoder will be used.
    static_layers: list[ResNetLayerConfig] = None # list of layer configurations for the static encoder. If None, no static encoder will be used.
    norm: str = "group"

    def __post_init__(self):
        # if a terrain encoder is used, validate the channel dims of the static encoder and decoder layers match for skip connections
        if self.static_layers is None:
            return

        emit_channels = [
            layer.out_channels for layer in self.static_layers
            if getattr(layer, 'emit_skip', False)
        ]

        if self.decoder_layers is not None:
            decoder_configs = self.decoder_layers
        else:
            decoder_configs = [
                layer.convert_to_decoder_layer() if hasattr(layer, 'convert_to_decoder_layer') else layer
                for layer in reversed(self.encoder_layers)
            ]

        receive_layers = [
            layer for layer in decoder_configs
            if getattr(layer, 'receives_skip', False)
        ]

        if len(emit_channels) != len(receive_layers):
            raise ValueError(
                f"static_layers emit {len(emit_channels)} skip(s) (emit_skip=True) but "
                f"decoder_layers receive {len(receive_layers)} skip(s) (receives_skip=True). "
                f"These counts must match exactly."
            )

        # decoder consumes skips coarse->fine, i.e. reversed relative to emission order
        skip_channels = list(reversed(emit_channels))
        running_channels = self.probabilistic_layer.channels
        skip_idx = 0

        for layer in decoder_configs:
            if not hasattr(layer, 'in_channels'):
                # ActivationLayerConfig, PatchAttentionLayerConfig (uses `channels` not in/out) etc.
                continue

            expected_in = running_channels
            skip_note = ""
            if getattr(layer, 'receives_skip', False):
                expected_in += skip_channels[skip_idx]
                skip_note = f" ({running_channels} from previous layer + {skip_channels[skip_idx]} from static skip)"
                skip_idx += 1

            if layer.in_channels != expected_in:
                raise ValueError(
                    f"Decoder layer {layer} declares in_channels={layer.in_channels}, but will "
                    f"receive {expected_in} channels at runtime{skip_note}. "
                    f"Fix in_channels on this layer config."
                )

            running_channels = layer.out_channels

@dataclass
class ActivationLayerConfig:
    activation: str = "GELU" # activation function

@dataclass
class ProbabilisticLayerConfig:
    latent_dim: int
    channels: int
    dim: int = 2
    logvar_clamp: tuple[float, float] = (-30.0, 20.0)

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

@dataclass
class ConvLayerConfig:
    dim: int # dimension of the convolution (1, 2, or 3)
    in_channels: int # number of input channels
    out_channels: int # number of output channels
    kernel_size: int # size of the convolution kernel
    norm: str = "group" # normalization method
    separable: bool = False # whether to use separable convolutions

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
    upsample_method: str = "nearest" # upsampling method (nearest, bilinear, trilinear, or transposed)
    sample_factor: int = 2 # upsampling factor
    receives_skip: bool = False # whether to receive a skip connection from the encoder

@dataclass
class PatchAttentionLayerConfig:
    dim: int # dimension of the input (1, 2, or 3)
    channels: int # number of input channels
    num_heads: int = 4 # number of attention heads
    patch_size: int = 1 # size of the patches for attention
    norm: str = "group" # normalization method
    dropout: float = 0.0 # dropout rate