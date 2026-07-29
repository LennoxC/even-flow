import pytest
import torch
from even_flow.module.autoencoder.VariationalAutoencoder import ConvolutionalVariationalAutoencoder
from even_flow.config.VariationalAutoencoderConfig import ConvolutionalVariationalAutoencoderConfig, DownsampleConvLayerConfig, UpsampleConvLayerConfig, ResNetLayerConfig, ConvLayerConfig

CONFIGS = [
    ConvolutionalVariationalAutoencoderConfig(
        input_dim=(3, 128, 128),
        encoder_layers=[
            DownsampleConvLayerConfig(dim=2, in_channels=3, out_channels=16, kernel_size=3, downsample_method="max"),
            DownsampleConvLayerConfig(dim=2, in_channels=16, out_channels=32, kernel_size=3, downsample_method="avg"),
            DownsampleConvLayerConfig(dim=2, in_channels=32, out_channels=64, kernel_size=3, downsample_method="max"),
        ],
        decoder_layers=[
            UpsampleConvLayerConfig(dim=2, in_channels=64, out_channels=32, kernel_size=3, upsample_method="nearest"),
            UpsampleConvLayerConfig(dim=2, in_channels=32, out_channels=16, kernel_size=3, upsample_method="bilinear"),
            UpsampleConvLayerConfig(dim=2, in_channels=16, out_channels=3, kernel_size=3, upsample_method="nearest"),
        ],
        activation="GELU",
        norm="group"),
    
    ConvolutionalVariationalAutoencoderConfig(
        input_dim=(3, 128, 128),
        encoder_layers=[
            ResNetLayerConfig(dim=2, in_channels=3, out_channels=16, kernel_size=3, activation="GELU", sampling="downsample", downsample_method="max"),
            ResNetLayerConfig(dim=2, in_channels=16, out_channels=32, kernel_size=3, activation="GELU", sampling="downsample", downsample_method="avg"),
            ResNetLayerConfig(dim=2, in_channels=32, out_channels=64, kernel_size=3, activation="GELU", sampling="downsample", downsample_method="max")
        ],
        decoder_layers=[
            ResNetLayerConfig(dim=2, in_channels=64, out_channels=32, kernel_size=3, activation="GELU", sampling="upsample", upsample_method="nearest"),
            ResNetLayerConfig(dim=2, in_channels=32, out_channels=16, kernel_size=3, activation="GELU", sampling="upsample", upsample_method="bilinear"),
            ResNetLayerConfig(dim=2, in_channels=16, out_channels=3, kernel_size=3, activation="Sigmoid", sampling="upsample", upsample_method="nearest")
        ],
        activation="ReLU",
        norm="batch"),
    
    ConvolutionalVariationalAutoencoderConfig(
        input_dim=(3, 128, 128),
        encoder_layers=[
            ResNetLayerConfig(dim=2, in_channels=3, out_channels=16, kernel_size=3, activation="GELU", sampling="downsample", downsample_method="max"),
            ResNetLayerConfig(dim=2, in_channels=16, out_channels=32, kernel_size=3, activation="GELU", sampling="downsample", downsample_method="avg"),
            ResNetLayerConfig(dim=2, in_channels=32, out_channels=64, kernel_size=3, activation="GELU", sampling="downsample", downsample_method="max"),
            ConvLayerConfig(dim=2, in_channels=64, out_channels=128, kernel_size=3)
        ],
        decoder_layers=[
            ConvLayerConfig(dim=2, in_channels=128, out_channels=64, kernel_size=3),
            ResNetLayerConfig(dim=2, in_channels=64, out_channels=32, kernel_size=3, activation="GELU", sampling="upsample", upsample_method="nearest"),
            ResNetLayerConfig(dim=2, in_channels=32, out_channels=16, kernel_size=3, activation="GELU", sampling="upsample", upsample_method="bilinear"),
            ResNetLayerConfig(dim=2, in_channels=16, out_channels=3, kernel_size=3, activation="Sigmoid", sampling="upsample", upsample_method="nearest")
        ],
        activation="GELU")
]

@pytest.mark.parametrize("model_config", CONFIGS)
def test_forward_conv_VariationalAutoencoder(model_config):
    config = model_config
    model = ConvolutionalVariationalAutoencoder(config)
    input_tensor = torch.randn(1, *config.input_dim)  # Batch size of 1
    output_tensor = model(input_tensor)
    assert output_tensor.shape == input_tensor.shape, f"Expected output shape {input_tensor.shape}, but got {output_tensor.shape}"