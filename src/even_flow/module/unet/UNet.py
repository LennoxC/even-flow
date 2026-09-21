import torch
from abc import ABC, abstractmethod
from even_flow.config import UNetConfig, ConvolutionalVariationalAutoencoderConfig, ResNetLayerConfig
from even_flow.module.conv.ConvLayer import UpsampleConvLayer, DownsampleConvLayer, ConvLayer
from even_flow.module.activation.ActivationLayer import ActivationLayer
from even_flow.module.conv.ResNetLayer import ResNetLayer
from even_flow.module.patch_attention.PatchAttentionBlock import PatchAttentionLayer
from even_flow.config import UpsampleConvLayerConfig
from even_flow.config import DownsampleConvLayerConfig, ConvLayerConfig, UpsampleConvLayerConfig, ActivationLayerConfig, PatchAttentionLayerConfig
from even_flow.module.vae.ProbabilisticLayer import ProbabilisticLatentEncoder, ProbabilisticLatentDecoder, reparameterize
from even_flow.module.autoencoder.StaticEncoder import StaticEncoder
from even_flow.module.autoencoder.Decoder import Decoder
from even_flow.module.model_base import UModelBase

class UNetBase(UModelBase):
    def __init__(self, config: UNetConfig):
        super().__init__()
        self.config = config
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()

    @abstractmethod
    def _build_encoder(self):
        pass

    @abstractmethod
    def _build_decoder(self):
        pass

    def forward(self, x):
        # Implement the forward pass for the UNet model
        encoder_outputs = self.encoder(x)
        decoder_output = self.decoder(encoder_outputs)
        return decoder_output

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)
    
    # ====== utility functions ======

    def print_model_summary(self, verbose=False):
        """
        Print a summary of the model, including the number of parameters and the percentage reduction in dimensionality.
        """
        print(f"Model Summary {'(Abbreviated)' if not verbose else '(Verbose)'}:")
        print(f"{'Pass verbose=True to see layer details' if not verbose else 'Pass verbose=False to see summary only'}")
        print('='*20)
        print(f"Number of parameters: {self.count_parameters()}")
        print(f"Number of trainable parameters: {self.count_trainable_parameters()}")
        print(f"Input dimensionality: {self.calculate_input_dimensionality()}: ({' x '.join(str(dim) for dim in self.config.input_dim)})")
        print(f"Latent dimensionality: {self.calculate_latent_dimensionality()}: ({' x '.join(str(dim) for dim in self.find_latent_dim_shape())})")
        print('='*20)
        if verbose:
            print("Encoder:")
            print(self.encoder)
            print('-'*20)
            print("Decoder:")
            print(self.decoder)

class UNet(UNetBase):
    def __init__(self, config: UNetConfig):
        super().__init__(config)

    def _build_encoder(self):
        layers = []

        for layer_config in self.config.encoder_layers:
            layers.append(self._build_layer(layer_config, self.config))

        return torch.nn.Sequential(*layers)

    def _build_decoder(self):
        configs = self.config.decoder_layers if self.config.decoder_layers is not None else reversed(self.config.encoder_layers)
        configs = [self._convert_to_decoder_layer(layer) for layer in configs] if self.config.decoder_layers is None else configs

        layers = []

        for layer_config in configs:
            layers.append(self._build_layer(layer_config, self.config))
        
        return torch.nn.Sequential(*layers)


    