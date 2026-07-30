from even_flow.config import *

"""
There are three components in this model:

- The encoder: takes in the input data (e.g. weather channels) and produces a latent representation (mean and log variance)
- The static encoder: takes in the static data (e.g. terrain channels) and produces a set of skip connections that are used in the decoder.
    - The static encoder doesn't contribute to the latent representation, which is reserved for dynamic input data.
    - At inference time, static data can be passed into the static encoder once to precompute the skip connections, which are then appended to the decoder network at the appropriate layers.
    - When using a flow matching model to generate latent representations, the encoder is not used at all. However the static encoder is still used to provide skip connections to the decoder.
- The decoder: takes in the latent representation and the skip connections from the static encoder and produces a reconstruction of the input data.

This model appends the first static skip connections in the first decoder layer, before the attention layes. This means that the attention layers in the decoder have access to a terrain representation.
"""

static_encoder_vae = ConvolutionalVariationalAutoencoderConfig(
        input_dim=(3, 128, 128),
        encoder_layers=[
            DownsampleConvLayerConfig(dim=2, in_channels=3, out_channels=16, kernel_size=3, downsample_method="max"),
            ActivationLayerConfig(activation="GELU"),
            DownsampleConvLayerConfig(dim=2, in_channels=16, out_channels=32, kernel_size=3, downsample_method="avg"),
            ActivationLayerConfig(activation="ReLU"),
            DownsampleConvLayerConfig(dim=2, in_channels=32, out_channels=64, kernel_size=3, downsample_method="max"),
            ActivationLayerConfig(activation="SiLU"),
            PatchAttentionLayerConfig(dim=2, channels=64, num_heads=4, patch_size=4, norm="group", dropout=0.0),
            ActivationLayerConfig(activation="ReLU"),
            PatchAttentionLayerConfig(dim=2, channels=64, num_heads=4, patch_size=8, norm="group", dropout=0.0),
            ActivationLayerConfig(activation="ReLU"),
        ],
        probabilistic_layer=ProbabilisticLayerConfig(latent_dim=64, channels=64),
        decoder_layers=[
            ConvLayerConfig(dim=2, in_channels=80, out_channels=64, kernel_size=3, receives_skip=True), # first skip connection arrives here (in channels 64 + 16 skip), before attention layers
            ActivationLayerConfig(activation="GELU"),
            PatchAttentionLayerConfig(dim=2, channels=64, num_heads=4, patch_size=8, norm="group", dropout=0.0),
            ActivationLayerConfig(activation="ReLU"),
            PatchAttentionLayerConfig(dim=2, channels=64, num_heads=4, patch_size=4, norm="group", dropout=0.0),
            ActivationLayerConfig(activation="ReLU"),
            UpsampleConvLayerConfig(dim=2, in_channels=64, out_channels=32, kernel_size=3, upsample_method="nearest"),
            ActivationLayerConfig(activation="GELU"),
            UpsampleConvLayerConfig(dim=2, in_channels=48, out_channels=16, kernel_size=3, upsample_method="bilinear", receives_skip=True), # second skip connection, (in channels 32 + 16 skip) after upsampling
            ActivationLayerConfig(activation="GELU"),
            UpsampleConvLayerConfig(dim=2, in_channels=24, out_channels=16, kernel_size=3, upsample_method="nearest", receives_skip=True), # final skip connection, after upsampling (in channels 16 + 8 skip)
            ActivationLayerConfig(activation="GELU"),
            ConvLayerConfig(dim=2, in_channels=16, out_channels=3, kernel_size=3),
            ActivationLayerConfig(activation="Sigmoid")
        ],
        static_dim=(3, 128, 128),
        static_layers=[
            ResNetLayerConfig(dim=2, in_channels=3, out_channels=8, kernel_size=3, activation="GELU", sampling="downsample", downsample_method="max", emit_skip=True),
            ResNetLayerConfig(dim=2, in_channels=8, out_channels=16, kernel_size=3, activation="GELU", sampling="downsample", downsample_method="avg", emit_skip=True),
            ResNetLayerConfig(dim=2, in_channels=16, out_channels=16, kernel_size=3, activation="GELU", sampling="downsample", downsample_method="max", emit_skip=True)
        ],
        activation="GELU",
        norm="group")