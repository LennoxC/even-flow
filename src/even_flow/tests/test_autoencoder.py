import pytest
import torch
from even_flow.module.autoencoder.VariationalAutoencoder import ConvolutionalVariationalAutoencoder
from even_flow.config.VariationalAutoencoderConfig import ConvolutionalVariationalAutoencoderConfig, DownsampleLayerConfig, UpsampleLayerConfig

CONFIGS = [
    ConvolutionalVariationalAutoencoderConfig(
        input_dim=(3, 128, 128),
        encoder_layers=[
            DownsampleLayerConfig(dim=2, in_channels=3, out_channels=16, kernel_size=3, activation="GELU", downsample_method="max"),
            DownsampleLayerConfig(dim=2, in_channels=16, out_channels=32, kernel_size=3, activation="GELU", downsample_method="avg"),
            DownsampleLayerConfig(dim=2, in_channels=32, out_channels=64, kernel_size=3, activation="GELU", downsample_method="max"),
        ],
        decoder_layers=[
            UpsampleLayerConfig(dim=2, in_channels=64, out_channels=32, kernel_size=3, activation="GELU", upsample_method="nearest"),
            UpsampleLayerConfig(dim=2, in_channels=32, out_channels=16, kernel_size=3, activation="GELU", upsample_method="bilinear"),
            UpsampleLayerConfig(dim=2, in_channels=16, out_channels=3, kernel_size=3, activation="Sigmoid", upsample_method="nearest"),
        ],
        activation="GELU",
        group_norm=False,
        batch_norm=False),
    
    ConvolutionalVariationalAutoencoderConfig(
        input_dim=(3, 128, 128),
        encoder_layers=[
            DownsampleLayerConfig(dim=2, in_channels=3, out_channels=16, kernel_size=3, activation="GELU", downsample_method="max"),
            DownsampleLayerConfig(dim=2, in_channels=16, out_channels=32, kernel_size=3, activation="GELU", downsample_method="avg"),
            DownsampleLayerConfig(dim=2, in_channels=32, out_channels=64, kernel_size=3, activation="GELU", downsample_method="max"),
        ],
        decoder_layers=[
            UpsampleLayerConfig(dim=2, in_channels=64, out_channels=32, kernel_size=3, activation="GELU", upsample_method="nearest"),
            UpsampleLayerConfig(dim=2, in_channels=32, out_channels=16, kernel_size=3, activation="GELU", upsample_method="bilinear"),
            UpsampleLayerConfig(dim=2, in_channels=16, out_channels=3, kernel_size=3, activation="Sigmoid", upsample_method="nearest"),
        ],
        activation="GELU",
        group_norm=True,
        batch_norm=False),
    
    ConvolutionalVariationalAutoencoderConfig(
        input_dim=(3, 128, 128),
        encoder_layers=[
            DownsampleLayerConfig(dim=2, in_channels=3, out_channels=16, kernel_size=3, activation="GELU", downsample_method="max"),
            DownsampleLayerConfig(dim=2, in_channels=16, out_channels=32, kernel_size=3, activation="GELU", downsample_method="avg"),
            DownsampleLayerConfig(dim=2, in_channels=32, out_channels=64, kernel_size=3, activation="GELU", downsample_method="max"),
        ],
        decoder_layers=[
            UpsampleLayerConfig(dim=2, in_channels=64, out_channels=32, kernel_size=3, activation="GELU", upsample_method="nearest"),
            UpsampleLayerConfig(dim=2, in_channels=32, out_channels=16, kernel_size=3, activation="GELU", upsample_method="bilinear"),
            UpsampleLayerConfig(dim=2, in_channels=16, out_channels=3, kernel_size=3, activation="Sigmoid", upsample_method="nearest"),
        ],
        activation="GELU",
        group_norm=False,
        batch_norm=True)
]

@pytest.mark.parametrize("model_config", CONFIGS)
def test_forward_conv_VariationalAutoencoder(model_config):
    config = model_config
    model = ConvolutionalVariationalAutoencoder(config)
    input_tensor = torch.randn(1, *config.input_dim)  # Batch size of 1
    output_tensor = model(input_tensor)
    assert output_tensor.shape == input_tensor.shape, f"Expected output shape {input_tensor.shape}, but got {output_tensor.shape}"