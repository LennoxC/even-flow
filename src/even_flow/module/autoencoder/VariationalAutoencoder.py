import torch
from abc import ABC, abstractmethod
from even_flow.config import VariationalAutoencoderConfig, ConvolutionalVariationalAutoencoderConfig, ResNetLayerConfig
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

class VariationalAutoencoderBase(UModelBase):
    """
    A base Variational Autoencoder implementation.
    - Use VariationalAutoencoderConfig to configure the model. 
        - This must contain at least an encoder_layers list and a probabilistic_layer config. 
        - The decoder_layers list is optional, and if not provided, the decoder will be the reverse of the encoder.
    - The encoder and decoder can be made from any combination of ConvLayer, ResNetLayer, UpsampleConvLayer, DownsampleConvLayer, and ActivationLayer.
    - The encoder outputs the mean and log variance of the latent space.
    - When the model is in .eval() mode, the mean of the latent space is used instead of sampling from it.
    - This will be used by the flow matching module to sample from the latent space.

    This is an abstract class and should not be instantiated directly. Use a subclass that implements the build_encoder and build_decoder methods.
    The only subclass currently implemented is ConvolutionalVariationalAutoencoder, which uses ConvLayer for the encoder and decoder. Other VariationalAutoencoder implementations may be implemented in the future.

    Parsing of the config file is handled in the build_encoder and build_decoder methods. This might get moved into a separate config parser class in the future.
    """
    def __init__(self, config: VariationalAutoencoderConfig):
        super().__init__()
        self.config = config
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

        if self.config.static_layers is not None:
            self.static_encoder = self.build_static_encoder()

    # UModelBase handles the _build_encoder and _build_decoder methods, along with other utility functions.
    # The _build_layer function is in the ModelBase class.

    @abstractmethod
    def build_static_encoder(self):
        pass

    def encode(self, x):
        return self.encoder(x)      # mean, log_var

    def sample(self, mean, log_var):
        # when model.eval() is called, the model is in evaluation mode and we should return the mean of the latent space instead of sampling from it
        if self.training:
            return reparameterize(mean, log_var)
        return mean
    
    def decode(self, z, static_skips=None):
        return self.decoder(z, static_skips)

    def forward(self, x, static=None):
        mean, log_var = self.encode(x)
        z = self.sample(mean, log_var)
        if self.config.static_layers is not None and static is not None:
            _, static_skips = self.static_encoder(static)
            x_recon = self.decoder(z, static_skips)
        else:
            x_recon = self.decoder(z, static_skips=[])
        return x_recon, mean, log_var

    def precompute_static_skips(self, static):
        if self.config.static_layers is not None:
            with torch.no_grad():
                _, static_skips = self.static_encoder(static)
                return static_skips
        else:
            raise ValueError("No static layers defined in the config. Cannot precompute static skips.")

    # ========== utility functions ==========

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
        print(f"Percentage reduction in dimensionality: {self.calculate_percentage_reduction():.2f}%")
        print('='*20)
        if verbose:
            print("Encoder:")
            print(self.encoder)
            print('-'*20)
            print("Decoder:")
            print(self.decoder)
            if self.config.static_layers is not None:
                print('-'*20)
                print("Static Encoder:")
                print(self.static_encoder)
        
    
class ConvolutionalVariationalAutoencoder(VariationalAutoencoderBase):
    """
    A basic Convolutional VariationalAutoencoder implementation.
    - Use ConvolutionalVariationalAutoencoderConfig to configure the model.
    - UNets are constructed for the encoder and decoder.
    - ResNet blocks or ConvLayer blocks can be used for the encoder and decoder.
    - If the ConvLayer blocks upsample or downsample, then use the corresponding UpsampleConvLayer or DownsampleConvLayer blocks for the decoder.
    """

    def __init__(self, config: ConvolutionalVariationalAutoencoderConfig):
        super().__init__(config)

    def build_encoder(self):
        layers = []
        for layer_config in self.config.encoder_layers:
            layers.append(self._build_layer(layer_config))

        probabilistic_encoder = ProbabilisticLatentEncoder(
            latent_dim=self.config.probabilistic_layer.latent_dim,
            channels=self.config.probabilistic_layer.channels,
            dim=self.config.probabilistic_layer.dim,
            logvar_clamp=self.config.probabilistic_layer.logvar_clamp
        )
        layers.append(probabilistic_encoder)

        return torch.nn.Sequential(*layers)

    def build_decoder(self):
        # if decoder layers are provided, use them. Otherwise, use the reverse of the encoder layers and convert to upsample layers
        configs = self.config.decoder_layers if self.config.decoder_layers is not None else reversed(self.config.encoder_layers)
        configs = [self._convert_to_decoder_layer(layer) for layer in configs] if self.config.decoder_layers is None else configs

        probabilistic_decoder = ProbabilisticLatentDecoder(
            latent_dim=self.config.probabilistic_layer.latent_dim,
            channels=self.config.probabilistic_layer.channels,
            dim=self.config.probabilistic_layer.dim
        )

        layers, receive_flags = [], []

        layers.append(probabilistic_decoder)
        receive_flags.append(False)
        for layer_config in configs:
            layers.append(self._build_layer(layer_config))
            receive_flags.append(getattr(layer_config, 'receives_skip', False))
        return Decoder(layers, receive_flags)

    def build_static_encoder(self):
        layers, emit_flags = [], []
        for layer_config in self.config.static_layers:
            layers.append(self._build_layer(layer_config))
            emit_flags.append(getattr(layer_config, 'emit_skip', False))
        return StaticEncoder(layers, emit_flags)