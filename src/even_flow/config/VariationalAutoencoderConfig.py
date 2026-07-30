from __future__ import annotations
from dataclasses import dataclass

@dataclass(kw_only=True)
class VariationalAutoencoderConfig:
    """
    Base configuration for a variational autoencoder. This class is intended to be subclassed for specific types of VAEs, such as convolutional VAEs or fully connected VAEs.
    
    Attributes:
        input_dim: tuple[int, int, int] - the dimensions of the input data (channels, height, width)
        output_dim: tuple[int, int, int] - the dimensions of the output data (channels, height, width). If None, output dims will be the same as input dims.
        activation: str - the default activation function to use if activation attributes are not specified in the layer configs (layer activations always override this default).
    """
    input_dim: tuple[int, int, int] # input dims [channels, height, width]
    output_dim: tuple[int, int, int] = None # output dims [channels, height, width]. If None, output dims will be the same as input dims

    activation: str = "GELU" # activation function

@dataclass
class ConvolutionalVariationalAutoencoderConfig(VariationalAutoencoderConfig):
    """
    Configuration for a convolutional variational autoencoder. This class extends the base VariationalAutoencoderConfig with additional attributes specific to convolutional VAEs.

    Attributes:
        encoder_layers: list[DownsampleConvLayerConfig | ConvLayerConfig | ResNetLayerConfig | ActivationLayerConfig | PatchAttentionLayerConfig] - a list of layer configurations for the encoder
        probabilistic_layer: ProbabilisticLayerConfig - the configuration for the probabilistic layer (latent moments)
        decoder_layers: list[UpsampleConvLayerConfig | ConvLayerConfig | ResNetLayerConfig | ActivationLayerConfig | PatchAttentionLayerConfig] - a list of layer configurations for the decoder. If None, the decoder will be the reverse of the encoder with nearest upsampling.
        static_dim: tuple[int, int, int] - the dimensions of the static input data (channels, height, width). If None, no static encoder will be used.
        static_layers: list[DownsampleConvLayerConfig | ConvLayerConfig | ResNetLayerConfig | ActivationLayerConfig | PatchAttentionLayerConfig] - a list of layer configurations for the static encoder. If None, no static encoder will be used.
        norm: str - the normalization method to use (e.g. "group", "batch", or None). This is the default normalization method to use if norm attributes are not specified in the layer configs (layer norms always override this default).
    """
    encoder_layers: list[DownsampleConvLayerConfig | ConvLayerConfig | ResNetLayerConfig | ActivationLayerConfig | PatchAttentionLayerConfig] # list of layer configurations
    probabilistic_layer: ProbabilisticLayerConfig # configuration for the probabilistic layer (latent moments)
    decoder_layers: list[UpsampleConvLayerConfig | ConvLayerConfig | ResNetLayerConfig | ActivationLayerConfig | PatchAttentionLayerConfig] = None # list of layer configurations. If None, decoder will be the reverse of the encoder with nearest upsampling
    static_dim: tuple[int, int, int] = None # input dims for the static encoder [channels, height, width]. If None, no static encoder will be used.
    static_layers: list[DownsampleConvLayerConfig | ConvLayerConfig | ResNetLayerConfig | ActivationLayerConfig | PatchAttentionLayerConfig] = None # list of layer configurations for the static encoder. If None, no static encoder will be used.
    norm: str = "group"

    def __post_init__(self):
        # if a static encoder is used, validate the channel dims of the static encoder and decoder layers match for skip connections
        if self.static_layers is None:
            return

        # if static layers are used but the static_dim is not specified, raise an error
        if self.static_dim is None:
            raise ValueError(
                "static_layers is not None but static_dim is None. "
                "If static_layers are used, static_dim must be specified."
            )

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

