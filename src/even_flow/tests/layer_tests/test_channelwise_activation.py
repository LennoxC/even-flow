import pytest
import torch
import torch.nn as nn
from even_flow.config import ChannelwiseActivationLayerConfig
from even_flow.module.model_base import ModelBase
from even_flow.module.activation.ChannelwiseActivation import ChannelwiseActivation

CONFIGS = [
    ChannelwiseActivationLayerConfig(activations=["ReLU"], dim=1),
    ChannelwiseActivationLayerConfig(activations=["Sigmoid"], dim=1),
    ChannelwiseActivationLayerConfig(activations=["Tanh"], dim=1),
    ChannelwiseActivationLayerConfig(activations=["ReLU", "Sigmoid"], dim=1),
    ChannelwiseActivationLayerConfig(activations=["ReLU", "Sigmoid", "Tanh"], dim=1),
    ChannelwiseActivationLayerConfig(activations=[nn.ReLU(), nn.Hardtanh(min_val=0, max_val=1)], dim=1),
]

@pytest.fixture(params=CONFIGS, ids=[f"ChannelwiseActivationLayerConfig_{i}" for i in range(len(CONFIGS))])
def channelwise_activation_layer_config(request):
    base = ModelBase()

    layer = base._build_layer(request.param)
    return layer


@pytest.mark.parametrize("config", CONFIGS)
@pytest.mark.layer
def test_channelwise_activation_layer_build(config):

    def channelwise_activation_check(layer, param, value):
        if param == "activations":
            assert isinstance(layer, ChannelwiseActivation)
            assert len(layer.activations) == len(value)

        if param == "dim":
            assert layer.dim == value

    def test_channelwise_activation_layer_forward(channelwise_activation_layer_config):
        layer = channelwise_activation_layer_config
        # Create a dummy input tensor with the appropriate number of channels
        if layer.dim == 1:
            x = torch.randn(2, len(layer.activations), 4, 4)  # batch_size=2, channels=len(activations), height=4, width=4
        elif layer.dim == 2:
            x = torch.randn(2, 3, len(layer.activations), 4)  # batch_size=2, channels=3, height=len(activations), width=4
        elif layer.dim == 3:
            x = torch.randn(2, 3, 4, len(layer.activations))  # batch_size=2, channels=3, height=4, width=len(activations)

        # Forward pass through the layer
        output = layer(x)

        # Check that the output shape matches the input shape
        assert output.shape == x.shape

        for i, act_layer in enumerate(layer.activations):
            layer_output = act_layer(x.select(layer.dim, i))
            if isinstance(act_layer, torch.nn.ReLU):
                assert torch.all(layer_output >= 0), "ReLU activation should produce non-negative outputs"
            if isinstance(act_layer, torch.nn.Sigmoid):
                assert torch.all((layer_output >= 0) & (layer_output <= 1)), "Sigmoid activation should produce outputs in the range [0, 1]"
            if isinstance(act_layer, torch.nn.Hardtanh):
                assert torch.all((layer_output >= 0) & (layer_output <= 1)), "Tanh activation should produce outputs in the range [0, 1]"

    base = ModelBase()
    layer = base._build_layer(config)
    assert layer is not None
    assert isinstance(layer, torch.nn.Module)

    for param in config.__dataclass_fields__:
        value = getattr(config, param)
        channelwise_activation_check(layer, param, value)

    test_channelwise_activation_layer_forward(layer)


    