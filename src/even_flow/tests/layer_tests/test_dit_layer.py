import pytest
import torch
from even_flow.config import DiTLayerConfig
from even_flow.module.model_base import ModelBase
from even_flow.module.dit.DiTLayer import DiTLayer

CONFIGS = [
    DiTLayerConfig(dim=2, channels=3, hidden_size=16, depth=4, num_heads=4, patch_size=2, time_conditioned=False),
    DiTLayerConfig(dim=2, channels=3, hidden_size=16, depth=4, num_heads=4, patch_size=2, time_conditioned=True, cond_dim=8),
    DiTLayerConfig(dim=2, channels=3, hidden_size=16, depth=4, num_heads=4, patch_size=2, time_conditioned=True, cond_dim=8, pos_embed="sincos+rope"),
]

_conv = {1: torch.nn.Conv1d, 2: torch.nn.Conv2d, 3: torch.nn.Conv3d}

@pytest.fixture(params=CONFIGS, ids=[f"DiTLayerConfig_{i}" for i in range(len(CONFIGS))])
def dit_layer_config(request):
    base = ModelBase()

    layer = base._build_layer(request.param)
    return layer


@pytest.mark.parametrize("config", CONFIGS)
@pytest.mark.layer
def test_dit_layer_build(config):

    def dit_layer_check(layer, param, value):
        if param == "dim": # check the patch embedding conv layer type based on dim
            assert isinstance(layer.patch_embed, _conv[value])
        if param == "channels":
            assert layer.patch_embed.in_channels == value
        if param == "hidden_size":
            expected_hidden_size = value if value is not None else layer.channels
            assert layer.patch_embed.out_channels == expected_hidden_size
        if param == "num_heads":
            for block in layer.blocks:
                assert block.attn.num_heads == value
        if param == "patch_size":
            assert layer.patch_embed.kernel_size == (value,) * layer.dim
            assert layer.patch_embed.stride == (value,) * layer.dim

    base = ModelBase()
    layer = base._build_layer(config)
    assert layer is not None
    assert isinstance(layer, torch.nn.Module)

    for param in config.__dataclass_fields__:
        value = getattr(config, param)
        dit_layer_check(layer, param, value)


    