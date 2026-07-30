from even_flow.config import *

"""
The most basic VAE model. 
- Encoder: a series of downsampling convolutional layers
- Probabilistic layer: to produce the latent representation
- Decoder: a series of upsampling convolutional layers. 

As no activation functions are included in the conv layers, the model manually specifies activation layers.

"""

basic_autoencoder = ConvolutionalVariationalAutoencoderConfig(
        input_dim=(3, 1024, 1024), 
        encoder_layers=[
            DownsampleConvLayerConfig(dim=2, in_channels=3, out_channels=16, kernel_size=3, sample_factor=2, downsample_method="max"),
            ActivationLayerConfig(activation="GELU"),
            DownsampleConvLayerConfig(dim=2, in_channels=16, out_channels=32, kernel_size=3, sample_factor=2, downsample_method="avg"),
            ActivationLayerConfig(activation="ReLU"),
            DownsampleConvLayerConfig(dim=2, in_channels=32, out_channels=64, kernel_size=3, sample_factor=2, downsample_method="max"),
            ActivationLayerConfig(activation="SiLU"),
            DownsampleConvLayerConfig(dim=2, in_channels=64, out_channels=128, kernel_size=3, sample_factor=2, downsample_method="max"),
            ActivationLayerConfig(activation="SiLU"),
        ],
        probabilistic_layer=ProbabilisticLayerConfig(latent_dim=128, channels=128),
        decoder_layers=[
            UpsampleConvLayerConfig(dim=2, in_channels=128, out_channels=64, kernel_size=3, sample_factor=2, upsample_method="nearest"),
            ActivationLayerConfig(activation="GELU"),
            UpsampleConvLayerConfig(dim=2, in_channels=64, out_channels=32, kernel_size=3, sample_factor=2, upsample_method="nearest"),
            ActivationLayerConfig(activation="GELU"),
            UpsampleConvLayerConfig(dim=2, in_channels=32, out_channels=16, kernel_size=3, sample_factor=2, upsample_method="bilinear"),
            ActivationLayerConfig(activation="GELU"),
            UpsampleConvLayerConfig(dim=2, in_channels=16, out_channels=3, kernel_size=3, sample_factor=2, upsample_method="nearest"),
            ActivationLayerConfig(activation="Sigmoid")
        ],
        activation="GELU",
        norm="group")