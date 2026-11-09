import pytest
import torch
from even_flow.config import ConvLayerConfig
from even_flow.module.model_base import ModelBase
from even_flow.module.conv.ConvLayer import ConvLayer

CONFIGS = [
    ConvLayerConfig(dim=2, in_channels=3, out_channels=16, kernel_size=3, pad_mode="reflect"),
    ConvLayerConfig(dim=2, in_channels=16, out_channels=32, kernel_size=3, pad_mode="same"),
    ConvLayerConfig(dim=2, in_channels=32, out_channels=64, kernel_size=3, pad_mode="valid"),
    ConvLayerConfig(dim=3, in_channels=32, out_channels=64, kernel_size=3, pad_mode="circular"),
    ConvLayerConfig(dim=3, in_channels=32, out_channels=32, kernel_size=3, pad_mode="same", separable=True)
]

@pytest.fixture(params=CONFIGS, ids=[f"ConvLayerConfig_{i}" for i in range(len(CONFIGS))])
def conv_layer_config(request):
    base = ModelBase()

    layer = base._build_layer(request.param)
    return layer


@pytest.mark.parametrize("config", CONFIGS)
@pytest.mark.layer
def test_conv_layer_build(config):

    def conv_layer_check(layer, param, value):
        conv = layer.conv
        if config.separable:
            assert isinstance(conv, torch.nn.Sequential)
            assert len(conv) == 2
            
        else: # check the conv layer parameters
            if param == "dim":
                if value == 2 and not getattr(config, "separable", False):
                    assert isinstance(conv, torch.nn.Conv2d)
                if value == 3 and not getattr(config, "separable", False):
                    assert isinstance(conv, torch.nn.Conv3d)
            
            if param == "in_channels":
                assert conv.in_channels == value
            if param == "out_channels":
                assert conv.out_channels == value
            if param == "kernel_size":
                expected = [value] * config.dim if isinstance(value, int) else value
                assert conv.kernel_size == tuple(expected), f"Expected kernel_size {expected}, got {conv.kernel_size}"
            if param == "pad_mode":
                assert conv.kernel_size == (value, value) if isinstance(value, int) else value, f"Expected kernel_size {value}, got {conv.kernel_size}"
            if param == "pad_mode" and isinstance(layer, ConvLayer): # pad modes: 'constant', 'reflect', 'replicate' or 'circular' (uses F.pad)
                if value == "constant":
                    x = torch.randn(1, conv.in_channels, 10, 10)
                    y = layer.pad(x)
                    assert torch.allclose(x, y)
                elif value == "reflect":
                    x = torch.randn(1, conv.in_channels, 10, 10)
                    y = layer.pad(x)
                    assert y.shape[2] == x.shape[2] + 2 and y.shape[3] == x.shape[3] + 2
                elif value == "replicate":
                    x = torch.randn(1, conv.in_channels, 10, 10)
                    y = layer.pad(x)
                    assert y.shape[2] == x.shape[2] + 2 and y.shape[3] == x.shape[3] + 2
                elif value == "circular":
                    x = torch.randn(1, conv.in_channels, 10, 10)
                    y = layer.pad(x)
                    assert y.shape[2] == x.shape[2] + 2 and y.shape[3] == x.shape[3] + 2

    base = ModelBase()
    layer = base._build_layer(config)
    assert layer is not None
    assert isinstance(layer, torch.nn.Module)

    for param in config.__dataclass_fields__:
        value = getattr(config, param)
        conv_layer_check(layer, param, value)


    