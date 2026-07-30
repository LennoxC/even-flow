from even_flow.config import *

"""
This is a demo of a 3D convolutional variational autoencoder.
- Encoder: a series of downsampling convolutional layers, with ResNet blocks and attention blocks
- Probabilistic layer: to produce the latent representation
- Decoder: a series of upsampling convolutional layers, with ResNet blocks and attention blocks

As no activation functions are included in the conv layers, the model manually specifies activation layers.
The model is designed to take in 3D data, such as temporal sequences of 2D frames, or a 3D atmospheric volume.
"""

dim_3_vae = ConvolutionalVariationalAutoencoderConfig(
        input_dim=(3, 32, 128, 128), # C, D, H, W
        encoder_layers=[
            ResNetLayerConfig(dim=3, in_channels=3, out_channels=16, kernel_size=3, activation="GELU", sampling="downsample", downsample_method="max"),
            ResNetLayerConfig(dim=3, in_channels=16, out_channels=32, kernel_size=3, activation="GELU", sampling="downsample", downsample_method="avg", separable=True),
            ResNetLayerConfig(dim=3, in_channels=32, out_channels=64, kernel_size=3, activation="GELU", sampling="downsample", downsample_method="max"),
            ConvLayerConfig(dim=3, in_channels=64, out_channels=128, kernel_size=3),
            PatchAttentionLayerConfig(dim=3, channels=128, num_heads=4, patch_size=4, norm="group", dropout=0.0),
            ActivationLayerConfig(activation="ReLU"),
            PatchAttentionLayerConfig(dim=3, channels=128, num_heads=4, patch_size=4, norm="group", dropout=0.0),
            ActivationLayerConfig(activation="ReLU")
        ],
        probabilistic_layer=ProbabilisticLayerConfig(dim=3, latent_dim=64, channels=128, logvar_clamp=(-30.0, 30.0)),
        decoder_layers=[
            PatchAttentionLayerConfig(dim=3, channels=128, num_heads=4, patch_size=4, norm="group", dropout=0.0),
            ActivationLayerConfig(activation="ReLU"),
            PatchAttentionLayerConfig(dim=3, channels=128, num_heads=4, patch_size=4, norm="group", dropout=0.0),
            ActivationLayerConfig(activation="ReLU"),
            ConvLayerConfig(dim=3, in_channels=128, out_channels=64, kernel_size=3),
            ResNetLayerConfig(dim=3, in_channels=64, out_channels=32, kernel_size=3, activation="GELU", sampling="upsample", upsample_method="nearest"),
            ResNetLayerConfig(dim=3, in_channels=32, out_channels=16, kernel_size=3, activation="GELU", sampling="upsample", upsample_method="trilinear", separable=True),
            ResNetLayerConfig(dim=3, in_channels=16, out_channels=3, kernel_size=3, activation="Sigmoid", sampling="upsample", upsample_method="nearest")
        ],
        activation="ReLU")