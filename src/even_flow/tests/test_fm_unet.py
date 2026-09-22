import pytest
import torch
from even_flow.module.unet.UNet import FlowMatchingUNet
from even_flow.config import TimeEncoderConfig, UNetConfig, ProbabilisticLayerConfig, DownsampleConvLayerConfig, UpsampleConvLayerConfig, ResNetLayerConfig, ConvLayerConfig, ActivationLayerConfig

CONFIGS = [
    UNetConfig(
            input_dim=(3, 128, 128),
            time_encoding=TimeEncoderConfig(encoding_strategy="Sinusoidal", time_encoding_dim=16, activation="GELU"),
            encoder_layers=[
                ResNetLayerConfig(dim=2, in_channels=3, out_channels=16, kernel_size=3, sampling="downsample", downsample_method="max", emit_skip=True),
                ResNetLayerConfig(dim=2, in_channels=16, out_channels=32, kernel_size=3, sampling="downsample", downsample_method="max", emit_skip=True),
                ResNetLayerConfig(dim=2, in_channels=32, out_channels=64, kernel_size=3, sampling="downsample", downsample_method="max", emit_skip=True)
            ],
            decoder_layers=[
                ResNetLayerConfig(dim=2, in_channels=128, out_channels=32, kernel_size=3, sampling="upsample", upsample_method="nearest", receives_skip=True),
                ResNetLayerConfig(dim=2, in_channels=64, out_channels=16, kernel_size=3, sampling="upsample", upsample_method="nearest", receives_skip=True),
                ResNetLayerConfig(dim=2, in_channels=32, out_channels=3, kernel_size=3, sampling="upsample", upsample_method="nearest", receives_skip=True)
            ],

            activation="GELU",
            norm="group"),

    UNetConfig(
            input_dim=(3, 128, 128),
            time_encoding=TimeEncoderConfig(encoding_strategy="Sinusoidal", time_encoding_dim=16, activation="GELU"),
            encoder_layers=[
                ResNetLayerConfig(dim=2, in_channels=3, out_channels=16, kernel_size=3, sampling="downsample", downsample_method="max"),
                ResNetLayerConfig(dim=2, in_channels=16, out_channels=32, kernel_size=3, sampling="downsample", downsample_method="max"),
                ResNetLayerConfig(dim=2, in_channels=32, out_channels=64, kernel_size=3, sampling="downsample", downsample_method="max")
            ],
            decoder_layers=[
                ResNetLayerConfig(dim=2, in_channels=64, out_channels=32, kernel_size=3, sampling="upsample", upsample_method="nearest"),
                ResNetLayerConfig(dim=2, in_channels=32, out_channels=16, kernel_size=3, sampling="upsample", upsample_method="nearest"),
                ResNetLayerConfig(dim=2, in_channels=16, out_channels=3, kernel_size=3, sampling="upsample", upsample_method="nearest")
            ],

            activation="GELU",
            norm="group")
]

@pytest.fixture(params=CONFIGS, ids=["FlowMatchingUNet", "NoSkipUnet"])
def model_and_input(request):
    config = request.param
    model = FlowMatchingUNet(config)
    x = torch.randn(1, *config.input_dim)
    t = torch.randn(1, 1)  # Random time input for the time encoding
    return model, x, t

@pytest.mark.fast
def test_forward_shape(model_and_input):
    model, x, t = model_and_input
    y = model(x, t)
    assert y.shape == x.shape

@pytest.mark.detailed
def test_gradients_flow(model_and_input):
    model, x, t = model_and_input
    y = model(x, t)
    y.sum().backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"{name} got no gradient at all (likely disconnected from graph)"
        assert p.grad.abs().sum() > 0, f"{name} gradient is entirely zero across all elements"

@pytest.mark.detailed
def test_model_summary(model_and_input):
    model, _, _ = model_and_input
    model.print_model_summary(verbose=True)