import pytest
import torch
from even_flow.module.autoencoder.VariationalAutoencoder import ConvolutionalVariationalAutoencoder
from even_flow.config.VariationalAutoencoderConfig import ConvolutionalVariationalAutoencoderConfig, DownsampleConvLayerConfig, UpsampleConvLayerConfig
from even_flow.utils.load_yaml import load_yaml

def test_forward_conv_VariationalAutoencoder():
    config = load_yaml(ConvolutionalVariationalAutoencoderConfig, "demo_arch/basic_autoencoder.yaml")
    model = ConvolutionalVariationalAutoencoder(config)
    input_tensor = torch.randn(1, *config.input_dim)  # Batch size of 1
    output_tensor = model(input_tensor)
    assert output_tensor.shape == input_tensor.shape, f"Expected output shape {input_tensor.shape}, but got {output_tensor.shape}"