import pytest
import torch
from even_flow.module.autoencoder.Autoencoder import Autoencoder
from even_flow.config.AutoencoderConfig import AutoencoderConfig

CONFIGS = [
    AutoencoderConfig(
        input_dim=(3, 32, 32),
        latent_dim=(3, 8),
        encoder_layers=[
            DownsampleLayerConfig(dim=2, in_channels=3, out_channels=16, kernel_size=3, activation="GeLU", downsample_method="max"),
            DownsampleLayerConfig(dim=2, in_channels=16, out_channels=32, kernel_size=3, activation="GeLU", downsample_method="avg"),
            DownsampleLayerConfig(dim=2, in_channels=32, out_channels=64, kernel_size=3, activation="GeLU", downsample_method="strided"),
        ],
        decoder_layers=[
            UpsampleLayerConfig(dim=2, in_channels=64, out_channels=32, kernel_size=3, activation="GeLU", upsample_method="nearest"),
            UpsampleLayerConfig(dim=2, in_channels=32, out_channels=16, kernel_size=3, activation="GeLU", upsample_method="bilinear"),
            UpsampleLayerConfig(dim=2, in_channels=16, out_channels=3, kernel_size=3, activation="Sigmoid", upsample_method="trilinear"),
        ],
        activation="GeLU",
        layer_norm=False,
        batch_norm=False)
]

@pytest.fixture(params=CONFIGS)
def test_forward_autoencoder(request):
    config = request.param
    model = Autoencoder(config)
    input_tensor = torch.randn(1, *config.input_dim)  # Batch size of 1
    output_tensor = model(input_tensor)
    assert output_tensor.shape == input_tensor.shape, f"Expected output shape {input_tensor.shape}, but got {output_tensor.shape}"