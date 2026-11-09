"""
Integration tests: DiTLayerConfig inside FlowMatchingUNet.

ASSUMPTION: I haven't seen UNetConfig / TimeEncoderConfig, so the constructor arguments below are
inferred from how UNetBase / TimeEmbedding read them (input_dim, encoder_layers, decoder_layers,
activation, time_encoding; time_encoding_dim, encoding_strategy, activation). Adjust if they differ.
"""
import pytest
import torch
from even_flow.config import (
    DiTLayerConfig, ResNetLayerConfig, TimeEncoderConfig, UNetConfig,
)
from even_flow.module.dit.DiTLayer import DiTLayer
from even_flow.module.unet.UNet import FlowMatchingUNet  # <- adjust to wherever your UNet file lives

TIME_DIM = 64


def _time_cfg():
    return TimeEncoderConfig(time_encoding_dim=TIME_DIM, encoding_strategy="Sinusoidal", activation="SiLU")


def _dit():
    # cond_dim is left as None on purpose: it is resolved from config.time_encoding.time_encoding_dim
    return DiTLayerConfig(dim=2, channels=32, hidden_size=64, depth=2, num_heads=4, patch_size=2)


def _down(i, o):
    return ResNetLayerConfig(dim=2, in_channels=i, out_channels=o, kernel_size=3, activation="GELU", sampling="downsample", downsample_method="max")


def _up(i, o):
    return ResNetLayerConfig(dim=2, in_channels=i, out_channels=o, kernel_size=3, activation="GELU", sampling="upsample", upsample_method="nearest")


CONFIGS = {
    # DiT at the end of the encoder (bottleneck), no DiT in the decoder
    "dit_end_of_encoder": UNetConfig(
        input_dim=(3, 64, 64),
        encoder_layers=[_down(3, 16), _down(16, 32), _dit()],
        decoder_layers=[_up(32, 16), _up(16, 3)],
        activation="GELU", time_encoding=_time_cfg()),
    # DiT at the start of the decoder
    "dit_start_of_decoder": UNetConfig(
        input_dim=(3, 64, 64),
        encoder_layers=[_down(3, 16), _down(16, 32)],
        decoder_layers=[_dit(), _up(32, 16), _up(16, 3)],
        activation="GELU", time_encoding=_time_cfg()),
}


@pytest.fixture(params=list(CONFIGS), ids=list(CONFIGS))
def model_and_input(request):
    config = CONFIGS[request.param]
    model = FlowMatchingUNet(config)
    x = torch.randn(2, *config.input_dim)
    t = torch.rand(2)
    return model, x, t


def _dit_layers(model):
    return [m for m in model.modules() if isinstance(m, DiTLayer)]


@pytest.mark.fast
def test_dit_is_built_with_time_conditioning(model_and_input):
    model, _, _ = model_and_input
    layers = _dit_layers(model)
    assert len(layers) == 1
    assert layers[0].cond_dim == TIME_DIM
    # the DiT handles time itself, so it must not also get an additive TimeInjection
    assert all(not hasattr(l, "in_channels") for l in layers)


@pytest.mark.fast
def test_forward_shape(model_and_input):
    model, x, t = model_and_input
    assert model(x, t).shape == x.shape


@pytest.mark.fast
def test_scalar_time(model_and_input):
    model, x, _ = model_and_input
    assert model(x, torch.tensor(0.3)).shape == x.shape


@pytest.mark.fast
def test_latent_shape_utilities_work_without_time(model_and_input):
    # calculate_latent_dimensionality() calls encode(x) with no t; must not crash for conditioned DiT layers
    model, _, _ = model_and_input
    assert model.calculate_latent_dimensionality() == 32 * 16 * 16


@pytest.mark.detailed
def test_gradients_reach_dit_and_time_embedding(model_and_input):
    model, x, t = model_and_input
    target = torch.randn_like(x)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

    # see test_dit_layer.test_gradients_flow for why adaLN-Zero needs several steps
    for _ in range(4):
        optimizer.zero_grad()
        loss = ((model(x, t) - target) ** 2).mean()
        loss.backward()
        optimizer.step()

    for name, p in model.named_parameters():
        if "blocks" in name or "final_adaLN" in name or "patch_embed" in name or "unpatch" in name:
            assert p.grad is not None, f"{name} got no gradient (disconnected from graph)"
            assert torch.isfinite(p.grad).all(), f"{name} has non-finite gradient"
            assert p.grad.abs().sum() > 0, f"{name} gradient is entirely zero"
    assert all(p.grad is not None for p in model.time_embedding.parameters())