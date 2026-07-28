import pytest
import torch
from even_flow.module.autoencoder.Autoencoder import ConvolutionalAutoencoder
from even_flow.config.AutoencoderConfig import ConvolutionalAutoencoderConfig, DownsampleLayerConfig, UpsampleLayerConfig
from even_flow.utils.load_yaml import load_yaml

def test_forward_conv_autoencoder():
    config = load_yaml(ConvolutionalAutoencoderConfig, "demo_arch/basic_autoencoder.yaml")
    model = ConvolutionalAutoencoder(config)
    input_tensor = torch.randn(1, *config.input_dim)  # Batch size of 1
    output_tensor = model(input_tensor)
    assert output_tensor.shape == input_tensor.shape, f"Expected output shape {input_tensor.shape}, but got {output_tensor.shape}"