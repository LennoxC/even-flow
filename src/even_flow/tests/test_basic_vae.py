import pytest
import torch
from even_flow.module.autoencoder.VariationalAutoencoder import ConvolutionalVariationalAutoencoder
from even_flow.config.VariationalAutoencoderConfig import ConvolutionalVariationalAutoencoderConfig, ProbabilisticLayerConfig, DownsampleConvLayerConfig, UpsampleConvLayerConfig, ResNetLayerConfig, ConvLayerConfig, ActivationLayerConfig

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
            UpsampleConvLayerConfig(dim=2, in_channels=64, out_channels=32, kernel_size=3, upsample_method="nearest"),
            ActivationLayerConfig(activation="GELU"),
            UpsampleConvLayerConfig(dim=2, in_channels=32, out_channels=16, kernel_size=3, upsample_method="bilinear"),
            ActivationLayerConfig(activation="GELU"),
            UpsampleConvLayerConfig(dim=2, in_channels=16, out_channels=3, kernel_size=3, upsample_method="nearest"),
            ActivationLayerConfig(activation="Sigmoid")
        ],
        activation="GELU",
        norm="group"),
    
    # test with ResNetBlock layers
    ConvolutionalVariationalAutoencoderConfig(
        input_dim=(3, 128, 128),
        encoder_layers=[
            ResNetLayerConfig(dim=2, in_channels=3, out_channels=16, kernel_size=3, activation="GELU", sampling="downsample", downsample_method="max"),
            ResNetLayerConfig(dim=2, in_channels=16, out_channels=32, kernel_size=3, activation="GELU", sampling="downsample", downsample_method="avg"),
            ResNetLayerConfig(dim=2, in_channels=32, out_channels=64, kernel_size=3, activation="GELU", sampling="downsample", downsample_method="max"),
            ConvLayerConfig(dim=2, in_channels=64, out_channels=128, kernel_size=3)
        ],
        probabilistic_layer=ProbabilisticLayerConfig(latent_dim=64, channels=128, dim=2, logvar_clamp=(-30.0, 30.0)),
        decoder_layers=[
            ConvLayerConfig(dim=2, in_channels=128, out_channels=64, kernel_size=3),
            ResNetLayerConfig(dim=2, in_channels=64, out_channels=32, kernel_size=3, activation="GELU", sampling="upsample", upsample_method="nearest"),
            ResNetLayerConfig(dim=2, in_channels=32, out_channels=16, kernel_size=3, activation="GELU", sampling="upsample", upsample_method="bilinear"),
            ResNetLayerConfig(dim=2, in_channels=16, out_channels=3, kernel_size=3, activation="Sigmoid", sampling="upsample", upsample_method="nearest")
        ],
        activation="ReLU"),

    # test with no decoder layers provided, so the decoder will be the reverse of the encoder
    ConvolutionalVariationalAutoencoderConfig(
        input_dim=(3, 128, 128),
        encoder_layers=[
            ResNetLayerConfig(dim=2, in_channels=3, out_channels=16, kernel_size=3, activation="GELU", sampling="downsample", downsample_method="max"),
            ResNetLayerConfig(dim=2, in_channels=16, out_channels=32, kernel_size=3, activation="GELU", sampling="downsample", downsample_method="avg"),
            DownsampleConvLayerConfig(dim=2, in_channels=32, out_channels=64, kernel_size=3, downsample_method="max"), # this should be converted to an UpsampleConvLayerConfig in the decoder
            ActivationLayerConfig(activation="ReLU"),
            ConvLayerConfig(dim=2, in_channels=64, out_channels=128, kernel_size=3), # the decoder will have a ConvLayerConfig with in_channels=128, out_channels=64
            ActivationLayerConfig(activation="GELU")
        ],
        probabilistic_layer=ProbabilisticLayerConfig(latent_dim=64, channels=128, dim=2, logvar_clamp=(-30.0, 30.0)),
        activation="GELU"),
]

@pytest.fixture(params=CONFIGS, ids=["conv_only", "resnet", "no_decoder_mirrors_encoder"])
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
    recon, mean, log_var = model(x)
    (recon.sum() + mean.sum() + log_var.sum()).backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"{name} got no gradient at all (likely disconnected from graph)"
        assert p.grad.abs().sum() > 0, f"{name} gradient is entirely zero across all elements"

@pytest.mark.detailed
def test_determinism(model_and_input):
    model, x = model_and_input
    model.eval()
    recon, mean, log_var = model(x)
    z = model.sample(mean, log_var)
    assert torch.allclose(z, mean), "Latent sample is not equal to mean when train=False. This indicates that the model is not deterministic during evaluation."