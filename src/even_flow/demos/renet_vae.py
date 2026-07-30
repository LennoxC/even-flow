from even_flow.config import *

"""
A basic VAE that uses ResNet blocks in the encoder and decoder. Two Patch Attention layers are used in the encoder and decoder on the coarse latent representation.

Each ResNet block contains the following:
- Conv2d layer
- Normalization layer (BatchNorm2d, GroupNorm, etc.)
- Activation layer (ReLU, GELU, etc.)
- Conv2d layer
- Normalization layer

"""

resnet_vae = ConvolutionalVariationalAutoencoderConfig(
        input_dim=(3, 128, 128),
        encoder_layers=[
            ResNetLayerConfig(dim=2, in_channels=3, out_channels=16, kernel_size=3, activation="GELU", sampling="downsample", downsample_method="max"),
            ResNetLayerConfig(dim=2, in_channels=16, out_channels=32, kernel_size=3, activation="GELU", sampling="downsample", downsample_method="avg"),
            ResNetLayerConfig(dim=2, in_channels=32, out_channels=64, kernel_size=3, activation="GELU", sampling="downsample", downsample_method="max"),
            PatchAttentionLayerConfig(dim=2, channels=64, num_heads=4, patch_size=4, norm="group", dropout=0.0),
            ActivationLayerConfig(activation="ReLU"),
            PatchAttentionLayerConfig(dim=2, channels=64, num_heads=4, patch_size=8, norm="group", dropout=0.0),
            ActivationLayerConfig(activation="ReLU"),
        ],
        probabilistic_layer=ProbabilisticLayerConfig(latent_dim=64, channels=64),
        decoder_layers=[
            PatchAttentionLayerConfig(dim=2, channels=64, num_heads=4, patch_size=8, norm="group", dropout=0.0),
            ActivationLayerConfig(activation="ReLU"),
            PatchAttentionLayerConfig(dim=2, channels=64, num_heads=4, patch_size=4, norm="group", dropout=0.0),
            ActivationLayerConfig(activation="ReLU"),
            ResNetLayerConfig(dim=2, in_channels=64, out_channels=32, kernel_size=3, activation="GELU", sampling="upsample", upsample_method="nearest"),
            ResNetLayerConfig(dim=2, in_channels=32, out_channels=16, kernel_size=3, activation="GELU", sampling="upsample", upsample_method="bilinear"),
            ResNetLayerConfig(dim=2, in_channels=16, out_channels=3, kernel_size=3, activation="Sigmoid", sampling="upsample", upsample_method="nearest")
        ],
        activation="ReLU",
        norm="batch")