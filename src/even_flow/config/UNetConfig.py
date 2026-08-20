from __future__ import annotations
from dataclasses import dataclass

@dataclass(kw_only=True)
class UNetConfig:
    """
    Configuration for a UNet model. This class defines the architecture of the UNet, including the encoder and decoder layers, as well as other parameters such as input/output dimensions and normalization methods.

    Attributes:
        input_dim: tuple[int, int, int] - the dimensions of the input data (channels, height, width)
        output_dim: tuple[int, int, int] - the dimensions of the output data (channels, height, width). If None, output dims will be the same as input dims.
        encoder_layers: list[DownsampleConvLayerConfig | ConvLayerConfig | ResNetLayerConfig | ActivationLayerConfig | PatchAttentionLayerConfig] - a list of layer configurations for the encoder
        decoder_layers: list[UpsampleConvLayerConfig | ConvLayerConfig | ResNetLayerConfig | ActivationLayerConfig | PatchAttentionLayerConfig] - a list of layer configurations for the decoder. If None, the decoder will be the reverse of the encoder with nearest upsampling.
        norm: str - the normalization method to use (e.g. "group", "batch", or None). This is the default normalization method to use if norm attributes are not specified in the layer configs (layer norms always override this default).
        activation: str - the activation function to use (e.g. "GELU", "ReLU", "SiLU", etc.).
    """
    input_dim: tuple[int, int, int] # input dims [channels, height, width]
    output_dim: tuple[int, int, int] = None # output dims [channels, height, width]. If None, output dims will be the same as input dims

    encoder_layers: list[DownsampleConvLayerConfig | ConvLayerConfig | ResNetLayerConfig | ActivationLayerConfig | PatchAttentionLayerConfig] # list of layer configurations
    decoder_layers: list[UpsampleConvLayerConfig | ConvLayerConfig | ResNetLayerConfig | ActivationLayerConfig | PatchAttentionLayerConfig] = None # list of layer configurations. If None, decoder will be the reverse of the encoder with nearest upsampling

    norm: str = "group"
    activation: str = "GELU" # activation function

