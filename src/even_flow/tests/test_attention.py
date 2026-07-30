import pytest
import torch
from even_flow.module.autoencoder.VariationalAutoencoder import ConvolutionalVariationalAutoencoder
from even_flow.config import ConvolutionalVariationalAutoencoderConfig, ProbabilisticLayerConfig, DownsampleConvLayerConfig, UpsampleConvLayerConfig, ResNetLayerConfig, ConvLayerConfig, ActivationLayerConfig, PatchAttentionLayerConfig

CONFIGS = [
    # test ResNetLayer layers + a single PatchAttentionBlock layer in the encoder and decoder
    ConvolutionalVariationalAutoencoderConfig(
        input_dim=(3, 128, 128),
        encoder_layers=[
            ResNetLayerConfig(dim=2, in_channels=3, out_channels=16, kernel_size=3, activation="GELU", sampling="downsample", downsample_method="max"),
            ResNetLayerConfig(dim=2, in_channels=16, out_channels=32, kernel_size=3, activation="GELU", sampling="downsample", downsample_method="avg"),
            ResNetLayerConfig(dim=2, in_channels=32, out_channels=64, kernel_size=3, activation="GELU", sampling="downsample", downsample_method="max"),
            PatchAttentionLayerConfig(dim=2, channels=64, num_heads=4, patch_size=4, norm="group", dropout=0.0),
            ActivationLayerConfig(activation="ReLU"),
        ],
        probabilistic_layer=ProbabilisticLayerConfig(latent_dim=64, channels=64, dim=2, logvar_clamp=(-30.0, 30.0)),
        decoder_layers=[
            PatchAttentionLayerConfig(dim=2, channels=64, num_heads=4, patch_size=4, norm="group", dropout=0.0),
            ActivationLayerConfig(activation="ReLU"),
            ResNetLayerConfig(dim=2, in_channels=64, out_channels=32, kernel_size=3, activation="GELU", sampling="upsample", upsample_method="nearest"),
            ResNetLayerConfig(dim=2, in_channels=32, out_channels=16, kernel_size=3, activation="GELU", sampling="upsample", upsample_method="bilinear"),
            ResNetLayerConfig(dim=2, in_channels=16, out_channels=3, kernel_size=3, activation="Sigmoid", sampling="upsample", upsample_method="nearest")
        ],
        activation="ReLU"),
    
    # test with two attention blocks in the encoder and decoder, with different patch sizes
    ConvolutionalVariationalAutoencoderConfig(
        input_dim=(3, 128, 128),
        encoder_layers=[
            ResNetLayerConfig(dim=2, in_channels=3, out_channels=16, kernel_size=3, activation="GELU", sampling="downsample", downsample_method="max"),
            ResNetLayerConfig(dim=2, in_channels=16, out_channels=32, kernel_size=3, activation="GELU", sampling="downsample", downsample_method="avg"),
            ResNetLayerConfig(dim=2, in_channels=32, out_channels=64, kernel_size=3, activation="GELU", sampling="downsample", downsample_method="max"),
            PatchAttentionLayerConfig(dim=2, channels=64, num_heads=4, patch_size=4, norm="group", dropout=0.0),
            ActivationLayerConfig(activation="ReLU"),
            PatchAttentionLayerConfig(dim=2, channels=64, num_heads=4, patch_size=8, norm="group", dropout=0.0),
            ActivationLayerConfig(activation="ReLU"),
        ],
        probabilistic_layer=ProbabilisticLayerConfig(latent_dim=64, channels=64, dim=2, logvar_clamp=(-30.0, 30.0)),
        decoder_layers=[
            PatchAttentionLayerConfig(dim=2, channels=64, num_heads=4, patch_size=8, norm="group", dropout=0.0),
            ActivationLayerConfig(activation="ReLU"),
            PatchAttentionLayerConfig(dim=2, channels=64, num_heads=4, patch_size=4, norm="group", dropout=0.0),
            ActivationLayerConfig(activation="ReLU"),
            ResNetLayerConfig(dim=2, in_channels=64, out_channels=32, kernel_size=3, activation="GELU", sampling="upsample", upsample_method="nearest"),
            ResNetLayerConfig(dim=2, in_channels=32, out_channels=16, kernel_size=3, activation="GELU", sampling="upsample", upsample_method="bilinear"),
            ResNetLayerConfig(dim=2, in_channels=16, out_channels=3, kernel_size=3, activation="Sigmoid", sampling="upsample", upsample_method="nearest")
        ],
        activation="ReLU"),

    # test 3D attention blocks in the encoder and decoder
    ConvolutionalVariationalAutoencoderConfig(
        input_dim=(3, 8, 128, 128), # C, D, H, W
        encoder_layers=[
            ResNetLayerConfig(dim=3, in_channels=3, out_channels=16, kernel_size=3, activation="GELU", sampling="downsample", downsample_method="max"),
            ResNetLayerConfig(dim=3, in_channels=16, out_channels=32, kernel_size=3, activation="GELU", sampling="downsample", downsample_method="avg", separable=True),
            DownsampleConvLayerConfig(dim=3, in_channels=32, out_channels=64, kernel_size=3, downsample_method="max"), # this should be converted to an UpsampleConvLayerConfig in the decoder
            ActivationLayerConfig(activation="ReLU"),
            ConvLayerConfig(dim=3, in_channels=64, out_channels=128, kernel_size=3), # the decoder will have a ConvLayerConfig with in_channels=128, out_channels=64
            ActivationLayerConfig(activation="GELU"),
            PatchAttentionLayerConfig(dim=3, channels=128, num_heads=4, patch_size=1, norm="group", dropout=0.0)
        ],
        probabilistic_layer=ProbabilisticLayerConfig(dim=3, latent_dim=64, channels=128, logvar_clamp=(-30.0, 30.0)),
        activation="GELU"),
]

@pytest.fixture(params=CONFIGS, ids=["basic_attention", "two_attention_blocks", "3d_attention_blocks"])
def model_and_input(request):
    config = request.param
    model = ConvolutionalVariationalAutoencoder(config)
    x = torch.randn(1, *config.input_dim)
    return model, x

@pytest.mark.fast
def test_forward_shape(model_and_input):
    model, x = model_and_input
    recon, mean, log_var = model(x)
    assert recon.shape == x.shape

@pytest.mark.detailed
def test_gradients_flow(model_and_input):
    model, x = model_and_input
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)  # much smaller lr

    # this test needs a few steps to ensure that gradients flow through the attention block, since it has a residual connection and could be skipped early in training
    for _ in range(2):
        optimizer.zero_grad()
        recon, mean, log_var = model(x)
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
    model, x = model_and_input
    model.eval()
    recon, mean, log_var = model(x)
    z = model.sample(mean, log_var)
    assert torch.allclose(z, mean), "Latent sample is not equal to mean when train=False. This indicates that the model is not deterministic during evaluation."