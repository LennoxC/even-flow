import torch
from abc import ABC, abstractmethod
from even_flow.config.VariationalAutoencoderConfig import VariationalAutoencoderConfig, ConvolutionalVariationalAutoencoderConfig, ResNetLayerConfig
from even_flow.module.conv.ConvLayer import ConvUpsampleLayer, ConvDownsampleLayer
from even_flow.module.conv.ResNetBlock import ResNetBlock
from even_flow.config.VariationalAutoencoderConfig import UpsampleConvLayerConfig
from even_flow.config.VariationalAutoencoderConfig import DownsampleConvLayerConfig, ConvLayerConfig, UpsampleConvLayerConfig

class VariationalAutoencoderBase(torch.nn.Module):
    """
    A base Variational Autoencoder implementation.
    - Use Variational AutoencoderConfig to configure the model.
    - ConvLayer is used for the encoder and decoder.
    - The encoder outputs the mean and log variance of the latent space.
    - This will be used by the flow matching module to sample from the latent space.
    """
    def __init__(self, config: VariationalAutoencoderConfig):
        super().__init__()
        self.config = config

        self.encoder = self.build_encoder()

        self.decoder = self.build_decoder()

    @abstractmethod
    def build_encoder(self):
        pass

    @abstractmethod
    def build_decoder(self):
        pass

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon

    
class ConvolutionalVariationalAutoencoder(VariationalAutoencoderBase):
    """
    A basic Convolutional VariationalAutoencoder implementation.
    - Use ConvolutionalVariationalAutoencoderConfig to configure the model.
    - UNets are constructed for the encoder and decoder.
    """

    def __init__(self, config: ConvolutionalVariationalAutoencoderConfig):
        super().__init__(config)

    def build_encoder(self):
        layers = []
        for layer_config in self.config.encoder_layers:
            layer = self._build_layer(layer_config, self.config)

        return torch.nn.Sequential(*layers)

    def build_decoder(self):
        layers = []

        configs = self.config.decoder_layers if self.config.decoder_layers is not None else reversed(self.config.encoder_layers)

        # if using the reverse of the encoder layers, switch the sampling method from downsample to upsample and vice versa
        configs = [UpsampleConvLayerConfig(**{**layer_config.__dict__, "sampling": "upsample" if layer_config.sampling == "downsample" else "downsample"}) 
                    if isinstance(layer_config, DownsampleConvLayerConfig) else layer_config for layer_config in configs]

        for layer_config in configs:
            layer = self._build_layer(layer_config, self.config)
        
        return torch.nn.Sequential(*layers)

    def _build_layer(self, layer_config, config):
        # global vs layer specific activation and norm
        activation = layer_config.activation if isinstance(layer_config, ResNetLayerConfig) and layer_config.activation is not None else self.config.activation
        norm = layer_config.norm if layer_config.norm is not None else self.config.norm

        if layer_config is ResNetLayerConfig:
            layers.append(ResNetBlock(
                dim=layer_config.dim,
                in_channels=layer_config.in_channels,
                out_channels=layer_config.out_channels,
                kernel_size=layer_config.kernel_size if hasattr(layer_config, 'kernel_size') else 3,
                norm=norm,
                separable=layer_config.separable if hasattr(layer_config, 'separable') else False,
                sampling=layer_config.sampling if hasattr(layer_config, 'sampling') else None,
                sample_factor=layer_config.sample_factor if hasattr(layer_config, 'sample_factor') else 2,
                upsample_method=layer_config.upsample_method if hasattr(layer_config, 'upsample_method') else "nearest",
                downsample_method=layer_config.downsample_method if hasattr(layer_config, 'downsample_method') else "strided",
                activation=activation,
                **{k: v for k, v in layer_config.__dict__.items() if k not in ['dim', 'in_channels', 'out_channels', 'kernel_size', 'norm', 'separable', 'sampling', 'sample_factor', 'upsample_method', 'downsample_method', 'activation']}
            ))

        if layer_config is UpsampleConvLayerConfig:
            layers.append(ConvUpsampleLayer(
                dim=layer_config.dim,
                in_channels=layer_config.in_channels,
                out_channels=layer_config.out_channels,
                kernel_size=layer_config.kernel_size if hasattr(layer_config, 'kernel_size') else 3,
                norm=norm,
                separable=layer_config.separable if hasattr(layer_config, 'separable') else False,
                upsample_method=layer_config.upsample_method if hasattr(layer_config, 'upsample_method') else "nearest",
                sample_factor=layer_config.sample_factor if hasattr(layer_config, 'sample_factor') else 2,
                activation=activation,
                **{k: v for k, v in layer_config.__dict__.items() if k not in ['dim', 'in_channels', 'out_channels', 'kernel_size', 'norm', 'separable', 'upsample_method', 'sample_factor', 'activation']}
            ))

        if layer_config is DownsampleConvLayerConfig:
            layers.append(ConvDownsampleLayer(
                dim=layer_config.dim,
                in_channels=layer_config.in_channels,
                out_channels=layer_config.out_channels,
                kernel_size=layer_config.kernel_size if hasattr(layer_config, 'kernel_size') else 3,
                norm=norm,
                separable=layer_config.separable if hasattr(layer_config, 'separable') else False,
                downsample_method=layer_config.downsample_method if hasattr(layer_config, 'downsample_method') else "strided",
                sample_factor=layer_config.sample_factor if hasattr(layer_config, 'sample_factor') else 2,
                activation=activation,
                **{k: v for k, v in layer_config.__dict__.items() if k not in ['dim', 'in_channels', 'out_channels', 'kernel_size', 'norm', 'separable', 'downsample_method', 'sample_factor', 'activation']}
            ))

        if layer_config is ConvLayerConfig:
            layers.append(ConvLayer(
                dim=layer_config.dim,
                in_channels=layer_config.in_channels,
                out_channels=layer_config.out_channels,
                kernel_size=layer_config.kernel_size if hasattr(layer_config, 'kernel_size') else 3,
                norm=norm,
                separable=layer_config.separable if hasattr(layer_config, 'separable') else False,
                activation=activation,
                **{k: v for k, v in layer_config.__dict__.items() if k not in ['dim', 'in_channels', 'out_channels', 'kernel_size', 'norm', 'separable', 'activation']}
            ))

    
