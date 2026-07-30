import pytest
import torch
from even_flow.module.autoencoder.VariationalAutoencoder import ConvolutionalVariationalAutoencoder
from even_flow.config import ConvolutionalVariationalAutoencoderConfig, ProbabilisticLayerConfig, DownsampleConvLayerConfig, UpsampleConvLayerConfig, ResNetLayerConfig, ConvLayerConfig, ActivationLayerConfig, PatchAttentionLayerConfig

CONFIGS = [
    # test with solely ConvLayer layers. These are the simplest layers and have no activation function.
    ConvolutionalVariationalAutoencoderConfig(
        input_dim=(3, 128, 128),
        encoder_layers=[
            DownsampleConvLayerConfig(dim=2, in_channels=3, out_channels=16, kernel_size=3, downsample_method="max"),
            ActivationLayerConfig(activation="GELU"),
            DownsampleConvLayerConfig(dim=2, in_channels=16, out_channels=32, kernel_size=3, downsample_method="avg"),
            ActivationLayerConfig(activation="ReLU"),
            DownsampleConvLayerConfig(dim=2, in_channels=32, out_channels=64, kernel_size=3, downsample_method="max"),
            ActivationLayerConfig(activation="SiLU"),
        ],
        probabilistic_layer=ProbabilisticLayerConfig(latent_dim=64, channels=64),
        decoder_layers=[
            UpsampleConvLayerConfig(dim=2, in_channels=128, out_channels=32, kernel_size=3, upsample_method="nearest", receives_skip=True),
            ActivationLayerConfig(activation="GELU"),
            UpsampleConvLayerConfig(dim=2, in_channels=64, out_channels=16, kernel_size=3, upsample_method="bilinear", receives_skip=True),
            ActivationLayerConfig(activation="GELU"),
            UpsampleConvLayerConfig(dim=2, in_channels=32, out_channels=3, kernel_size=3, upsample_method="nearest", receives_skip=True),
            ActivationLayerConfig(activation="Sigmoid")
        ],
        static_dim=(3, 128, 128),
        static_layers=[ # the decoder will 
            ResNetLayerConfig(dim=2, in_channels=3, out_channels=16, kernel_size=3, activation="GELU", sampling="downsample", downsample_method="max", emit_skip=True),
            ResNetLayerConfig(dim=2, in_channels=16, out_channels=32, kernel_size=3, activation="GELU", sampling="downsample", downsample_method="avg", emit_skip=True),
            ResNetLayerConfig(dim=2, in_channels=32, out_channels=64, kernel_size=3, activation="GELU", sampling="downsample", downsample_method="max", emit_skip=True)
        ],
        activation="GELU",
        norm="group"),

    # test with attention layers in the encoder and decoder
    # uses a more complex skip connection pattern
    ConvolutionalVariationalAutoencoderConfig(
        input_dim=(3, 128, 128),
        encoder_layers=[
            DownsampleConvLayerConfig(dim=2, in_channels=3, out_channels=16, kernel_size=3, downsample_method="max"),
            ActivationLayerConfig(activation="GELU"),
            DownsampleConvLayerConfig(dim=2, in_channels=16, out_channels=32, kernel_size=3, downsample_method="avg"),
            ActivationLayerConfig(activation="ReLU"),
            DownsampleConvLayerConfig(dim=2, in_channels=32, out_channels=64, kernel_size=3, downsample_method="max"),
            ActivationLayerConfig(activation="SiLU"),
            PatchAttentionLayerConfig(dim=2, channels=64, num_heads=4, patch_size=4, norm="group", dropout=0.0),
            ActivationLayerConfig(activation="ReLU"),
            PatchAttentionLayerConfig(dim=2, channels=64, num_heads=4, patch_size=8, norm="group", dropout=0.0),
            ActivationLayerConfig(activation="ReLU"),
        ],
        probabilistic_layer=ProbabilisticLayerConfig(latent_dim=64, channels=64),
        decoder_layers=[
            ConvLayerConfig(dim=2, in_channels=80, out_channels=64, kernel_size=3, receives_skip=True), # first skip connection arrives here (in channels 64 + 16 skip), before attention layers
            ActivationLayerConfig(activation="GELU"),
            PatchAttentionLayerConfig(dim=2, channels=64, num_heads=4, patch_size=8, norm="group", dropout=0.0),
            ActivationLayerConfig(activation="ReLU"),
            PatchAttentionLayerConfig(dim=2, channels=64, num_heads=4, patch_size=4, norm="group", dropout=0.0),
            ActivationLayerConfig(activation="ReLU"),
            UpsampleConvLayerConfig(dim=2, in_channels=64, out_channels=32, kernel_size=3, upsample_method="nearest"),
            ActivationLayerConfig(activation="GELU"),
            UpsampleConvLayerConfig(dim=2, in_channels=48, out_channels=16, kernel_size=3, upsample_method="bilinear", receives_skip=True), # second skip connection, (in channels 32 + 16 skip) after upsampling
            ActivationLayerConfig(activation="GELU"),
            UpsampleConvLayerConfig(dim=2, in_channels=24, out_channels=16, kernel_size=3, upsample_method="nearest", receives_skip=True), # final skip connection, after upsampling (in channels 16 + 8 skip)
            ActivationLayerConfig(activation="GELU"),
            ConvLayerConfig(dim=2, in_channels=16, out_channels=3, kernel_size=3),
            ActivationLayerConfig(activation="Sigmoid")
        ],
        static_dim=(3, 128, 128),
        static_layers=[
            ResNetLayerConfig(dim=2, in_channels=3, out_channels=8, kernel_size=3, activation="GELU", sampling="downsample", downsample_method="max", emit_skip=True),
            ResNetLayerConfig(dim=2, in_channels=8, out_channels=16, kernel_size=3, activation="GELU", sampling="downsample", downsample_method="avg", emit_skip=True),
            ResNetLayerConfig(dim=2, in_channels=16, out_channels=16, kernel_size=3, activation="GELU", sampling="downsample", downsample_method="max", emit_skip=True)
        ],
        activation="GELU",
        norm="group"),
]

@pytest.fixture(params=CONFIGS, ids=["minimal_static_encoder", "static_encoder_with_attention"])
def model_and_input(request):
    config = request.param
    model = ConvolutionalVariationalAutoencoder(config)
    x = torch.randn(1, *config.input_dim)
    static = torch.randn(1, *config.static_dim)
    return model, x, static

@pytest.mark.fast
def test_forward_shape(model_and_input):
    model, x, static = model_and_input
    recon, mean, log_var = model(x, static)
    assert recon.shape == x.shape

@pytest.mark.detailed
def test_gradients_flow(model_and_input):
    model, x, static = model_and_input
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)  # much smaller lr

    # this test needs a few steps to ensure that gradients flow through the attention block, since it has a residual connection and could be skipped early in training
    for _ in range(2):
        optimizer.zero_grad()
        recon, mean, log_var = model(x, static)
        loss = recon.mean() + mean.mean() + log_var.mean()  # mean, not sum
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # guard against blow-ups
        optimizer.step()

        # fail fast and clearly if anything has gone non-finite, rather than
        # letting NaN silently masquerade as "zero gradient" several steps later
        for name, p in model.named_parameters():
            assert torch.isfinite(p).all(), f"{name} became non-finite (NaN/Inf) during training"

    for name, p in model.named_parameters():
        assert p.grad is not None, f"{name} got no gradient at all (likely disconnected from graph)"
        assert torch.isfinite(p.grad).all(), f"{name} has non-finite gradient"
        assert p.grad.abs().sum() > 0, f"{name} gradient is entirely zero across all elements"

@pytest.mark.detailed
def test_determinism(model_and_input):
    model, x, static = model_and_input
    model.eval()
    recon, mean, log_var = model(x, static)
    z = model.sample(mean, log_var)
    assert torch.allclose(z, mean), "Latent sample is not equal to mean when train=False. This indicates that the model is not deterministic during evaluation."

@pytest.mark.detailed
def test_inference_time(model_and_input):
    model, x, static = model_and_input
    model.eval()
    with torch.no_grad():
        static_skips = model.precompute_static_skips(static) # compute the static skips once
        z = torch.randn(1, 64, 16, 16) # B, C, W, H sample a random latent vector (in practice, this will come from the flow model)
        
        out = model.decode(z, static_skips) # decode using the precomputed static skips

        assert out.shape == x.shape, f"Expected output shape {x.shape}, but got {out.shape}"

@pytest.mark.detailed
def test_precompute_equivalence(model_and_input):
    model, x, static = model_and_input
    model.eval()
    with torch.no_grad():
        mean, log_var = model.encode(x)
        z = model.sample(mean, log_var)

        out_fused, _, _ = model(x, static)
        skips = model.precompute_static_skips(static)
        out_cached = model.decode(z, skips)

    assert torch.allclose(out_fused, out_cached, atol=1e-6)