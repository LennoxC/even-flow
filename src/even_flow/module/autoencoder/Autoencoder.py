import torch
from abc import ABC, abstractmethod
from even_flow.config.AutoencoderConfig import AutoencoderConfig, ConvolutionalAutoencoderConfig
from even_flow.module.conv.ConvLayer import ConvUpsampleLayer, ConvDownsampleLayer
from even_flow.config.AutoencoderConfig import UpsampleLayerConfig

class AutoencoderBase(torch.nn.Module):
    """
    A base Autoencoder implementation.
    - Use AutoencoderConfig to configure the model.
    - ConvLayer is used for the encoder and decoder.
    - The encoder outputs the mean and log variance of the latent space.
    - This will be used by the flow matching module to sample from the latent space.
    """
    def __init__(self, config: AutoencoderConfig):
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

    
class ConvolutionalAutoencoder(AutoencoderBase):
    """
    A basic Convolutional Autoencoder implementation.
    - Use ConvolutionalAutoencoderConfig to configure the model.
    - UNets are constructed for the encoder and decoder.
    """

    def __init__(self, config: ConvolutionalAutoencoderConfig):
        super().__init__(config)

    def build_encoder(self):
        layers = []
        for layer_config in self.config.encoder_layers:

            activation = layer_config.activation if layer_config.activation is not None else self.config.activation

            layers.append(ConvDownsampleLayer(dim=layer_config.dim,
                                              in_channels=layer_config.in_channels,
                                              out_channels=layer_config.out_channels,
                                              kernel_size=layer_config.kernel_size,
                                              activation=activation,
                                              downsample_method=layer_config.downsample_method))

            norm = self._norm(layer_config)
            if norm is not None:
                layers.append(norm)
        
        return torch.nn.Sequential(*layers)

    def build_decoder(self):
        layers = []

        configs = self.config.decoder_layers if self.config.decoder_layers is not None else reversed(self.config.encoder_layers)

        for layer_config in configs:
            activation = layer_config.activation if layer_config.activation is not None else self.config.activation
            upsample_method = layer_config.upsample_method if isinstance(layer_config, UpsampleLayerConfig) else "nearest"

            layers.append(ConvUpsampleLayer(dim=layer_config.dim,
                                            in_channels=layer_config.out_channels,
                                            out_channels=layer_config.in_channels,
                                            kernel_size=layer_config.kernel_size,
                                            activation=activation,
                                            upsample_method=upsample_method))
            
            norm = self._norm(layer_config)
            if norm is not None:
                layers.append(norm)
        
        return torch.nn.Sequential(*layers)

    def _norm(self, layer_config):
        if self.config.layer_norm:
            
            return torch.nn.LayerNorm([layer_config.out_channels, ])
        elif self.config.batch_norm:
            return getattr(torch.nn, f"BatchNorm{layer_config.dim}d")(layer_config.out_channels)
        else:
            return None

    
