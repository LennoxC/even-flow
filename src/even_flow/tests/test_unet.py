import pytest
import torch
from even_flow.module.unet.UNet import UNet
from even_flow.config import UNetConfig, ProbabilisticLayerConfig, DownsampleConvLayerConfig, UpsampleConvLayerConfig, ResNetLayerConfig, ConvLayerConfig, ActivationLayerConfig

CONFIGS = [
    # test with solely ConvLayer layers. These are the simplest layers and have no activation function.
    UNetConfig(
        input_dim=(3, 128, 128),
        encoder_layers=[
            DownsampleConvLayerConfig(dim=2, in_channels=3, out_channels=16, kernel_size=3, downsample_method="max"),
            ActivationLayerConfig(activation="GELU"),
            DownsampleConvLayerConfig(dim=2, in_channels=16, out_channels=32, kernel_size=3, downsample_method="avg"),
            ActivationLayerConfig(activation="ReLU"),
            DownsampleConvLayerConfig(dim=2, in_channels=32, out_channels=64, kernel_size=3, downsample_method="max"),
            ActivationLayerConfig(activation="SiLU"),
        ],
        decoder_layers=[
            UpsampleConvLayerConfig(dim=2, in_channels=64, out_channels=32, kernel_size=3, upsample_method="nearest"),
            ActivationLayerConfig(activation="GELU"),
            UpsampleConvLayerConfig(dim=2, in_channels=32, out_channels=16, kernel_size=3, upsample_method="bilinear"),
            ActivationLayerConfig(activation="GELU"),
            UpsampleConvLayerConfig(dim=2, in_channels=16, out_channels=3, kernel_size=3, upsample_method="nearest"),
            ActivationLayerConfig(activation="Sigmoid")
        ],
        activation="GELU",
        norm="group"),

    UNetConfig(
            input_dim=(3, 128, 128),
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
            encoder_layers=[
                ResNetLayerConfig(dim=2, in_channels=3, out_channels=16, kernel_size=3, sampling="downsample", downsample_method="max", emit_skip=True),
                ConvLayerConfig(dim=2, in_channels=16, out_channels=16, kernel_size=3),
                ActivationLayerConfig(activation="ReLU"),
                ResNetLayerConfig(dim=2, in_channels=16, out_channels=32, kernel_size=3, sampling="downsample", downsample_method="max", emit_skip=True),
                ResNetLayerConfig(dim=2, in_channels=32, out_channels=64, kernel_size=3, sampling="downsample", downsample_method="max", emit_skip=True)
            ],
            decoder_layers=[
                ResNetLayerConfig(dim=2, in_channels=128, out_channels=32, kernel_size=3, sampling="upsample", upsample_method="nearest", receives_skip=True),
                ConvLayerConfig(dim=2, in_channels=32, out_channels=32, kernel_size=3),
                ActivationLayerConfig(activation="ReLU"),
                ResNetLayerConfig(dim=2, in_channels=64, out_channels=16, kernel_size=3, sampling="upsample", upsample_method="nearest", receives_skip=True),
                ResNetLayerConfig(dim=2, in_channels=32, out_channels=3, kernel_size=3, sampling="upsample", upsample_method="nearest", receives_skip=True)
            ],
            activation="GELU",
            norm="group")
]

@pytest.fixture(params=CONFIGS, ids=["basic_unet_no_skip", "resnet_unet", "resnet_conv_unet"])
def model_and_input(request):
    config = request.param
    model = UNet(config)
    x = torch.randn(1, *config.input_dim)
    return model, x

@pytest.mark.fast
def test_forward_shape(model_and_input):
    model, x = model_and_input
    y = model(x)
    assert y.shape == x.shape

@pytest.mark.detailed
def test_gradients_flow(model_and_input):
    model, x = model_and_input
    y = model(x)
    y.sum().backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"{name} got no gradient at all (likely disconnected from graph)"
        assert p.grad.abs().sum() > 0, f"{name} gradient is entirely zero across all elements"

@pytest.mark.detailed
def test_model_summary(model_and_input):
    model, _ = model_and_input
    model.print_model_summary(verbose=True)