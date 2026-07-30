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

class VariationalAutoencoderBase(torch.nn.Module):
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

    @abstractmethod
    def build_encoder(self):
        pass

    @abstractmethod
    def build_decoder(self):
        pass

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

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def count_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def calculate_latent_dimensionality(self):
        """
        Calculate the dimensionality of the latent space based on the encoder output.
        This is useful for determining the size of the latent space for the flow model.
        """
        # create a dummy input tensor with the same shape as the input data
        dummy_input = torch.randn(1, *self.config.input_dim)
        mean, log_var = self.encode(dummy_input)
        return torch.prod(torch.tensor(mean.shape[1:])).item()  # exclude batch dimension

    def find_latent_dim_shape(self):
        """
        Find the shape of the latent space based on the encoder output.
        This is useful for determining the shape of the latent space for the flow model.
        """
        # create a dummy input tensor with the same shape as the input data
        dummy_input = torch.randn(1, *self.config.input_dim)
        mean, log_var = self.encode(dummy_input)
        return mean.shape[1:]  # exclude batch dimension

    def calculate_input_dimensionality(self):
        """
        Calculate the number of elements in the input tensor based on the input dimensions specified in the config.
        """
        return torch.prod(torch.tensor(self.config.input_dim)).item()  # exclude batch dimension

    def calculate_percentage_reduction(self):
        """
        Calculate the percentage reduction in dimensionality from the input to the latent space.
        This is useful for determining how much information is being compressed by the encoder.
        """
        input_dimensionality = self.calculate_input_dimensionality()
        latent_dimensionality = self.calculate_latent_dimensionality()
        return (1 - (latent_dimensionality / input_dimensionality)) * 100

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
            layers.append(self._build_layer(layer_config, self.config))

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
            layers.append(self._build_layer(layer_config, self.config))
            receive_flags.append(getattr(layer_config, 'receives_skip', False))
        return Decoder(layers, receive_flags)

    def build_static_encoder(self):
        layers, emit_flags = [], []
        for layer_config in self.config.static_layers:
            layers.append(self._build_layer(layer_config, self.config))
            emit_flags.append(getattr(layer_config, 'emit_skip', False))
        return StaticEncoder(layers, emit_flags)

    def _build_layer(self, layer_config, config):
        """
        Converts from config files into actual layer objects.
        """

        if isinstance(layer_config, ActivationLayerConfig):
            return ActivationLayer(activation=layer_config.activation)

        # global vs layer specific activation and norm
        activation = layer_config.activation if isinstance(layer_config, ResNetLayerConfig) and layer_config.activation is not None else self.config.activation
        norm = layer_config.norm if layer_config.norm is not None else self.config.norm

        if isinstance(layer_config, ResNetLayerConfig):
            return ResNetLayer(
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
                **{k: v for k, v in layer_config.__dict__.items() if k not in ['dim', 'in_channels', 'out_channels', 'kernel_size', 'norm', 'separable', 'sampling', 'sample_factor', 'upsample_method', 'downsample_method', 'activation', 'emit_skip', 'receives_skip']}
            )

        if isinstance(layer_config, UpsampleConvLayerConfig):
            return UpsampleConvLayer(
                dim=layer_config.dim,
                in_channels=layer_config.in_channels,
                out_channels=layer_config.out_channels,
                kernel_size=layer_config.kernel_size if hasattr(layer_config, 'kernel_size') else 3,
                norm=norm,
                separable=layer_config.separable if hasattr(layer_config, 'separable') else False,
                upsample_method=layer_config.upsample_method if hasattr(layer_config, 'upsample_method') else "nearest",
                sample_factor=layer_config.sample_factor if hasattr(layer_config, 'sample_factor') else 2,
                **{k: v for k, v in layer_config.__dict__.items() if k not in ['dim', 'in_channels', 'out_channels', 'kernel_size', 'norm', 'separable', 'upsample_method', 'sample_factor', 'emit_skip', 'receives_skip']}
            )

        if isinstance(layer_config, DownsampleConvLayerConfig):
            return DownsampleConvLayer(
                dim=layer_config.dim,
                in_channels=layer_config.in_channels,
                out_channels=layer_config.out_channels,
                kernel_size=layer_config.kernel_size if hasattr(layer_config, 'kernel_size') else 3,
                norm=norm,
                separable=layer_config.separable if hasattr(layer_config, 'separable') else False,
                downsample_method=layer_config.downsample_method if hasattr(layer_config, 'downsample_method') else "strided",
                sample_factor=layer_config.sample_factor if hasattr(layer_config, 'sample_factor') else 2,
                **{k: v for k, v in layer_config.__dict__.items() if k not in ['dim', 'in_channels', 'out_channels', 'kernel_size', 'norm', 'separable', 'downsample_method', 'sample_factor', 'emit_skip', 'receives_skip']}
            )

        if isinstance(layer_config, ConvLayerConfig):
            return ConvLayer(
                dim=layer_config.dim,
                in_channels=layer_config.in_channels,
                out_channels=layer_config.out_channels,
                kernel_size=layer_config.kernel_size if hasattr(layer_config, 'kernel_size') else 3,
                norm=norm,
                separable=layer_config.separable if hasattr(layer_config, 'separable') else False,
                **{k: v for k, v in layer_config.__dict__.items() if k not in ['dim', 'in_channels', 'out_channels', 'kernel_size', 'norm', 'separable', 'emit_skip', 'receives_skip']}
            )

        if isinstance(layer_config, PatchAttentionLayerConfig):
            return PatchAttentionLayer(
                dim=layer_config.dim,
                channels=layer_config.channels,
                num_heads=layer_config.num_heads if hasattr(layer_config, 'num_heads') else 4,
                patch_size=layer_config.patch_size if hasattr(layer_config, 'patch_size') else 1,
                norm=layer_config.norm if hasattr(layer_config, 'norm') else "group",
                dropout=layer_config.dropout if hasattr(layer_config, 'dropout') else 0.0,
                **{k: v for k, v in layer_config.__dict__.items() if k not in ['dim', 'channels', 'num_heads', 'patch_size', 'norm', 'dropout', 'emit_skip', 'receives_skip']}
            )

        raise ValueError(f"Unknown layer config type: {layer_config}")
        return None

    def _convert_to_decoder_layer(self, layer_config):
        """
        Convert an encoder layer config to a decoder layer config.
        - For ConvLayerConfig, flip the in_channels and out_channels (handled by the convert_to_decoder_layer method).
        - For DownsampleConvLayerConfig, convert to UpsampleConvLayerConfig and flip the in_channels and out_channels (handled by the convert_to_decoder_layer method).
        - For ResNetLayerConfig, convert to the corresponding decoder layer config using the convert_to_decoder_layer method (handled by the convert_to_decoder_layer method).
        """
        if isinstance(layer_config, ConvLayerConfig):
            return layer_config.convert_to_decoder_layer()
        elif isinstance(layer_config, DownsampleConvLayerConfig):
            return layer_config.convert_to_decoder_layer()
        elif isinstance(layer_config, ResNetLayerConfig):
            return layer_config.convert_to_decoder_layer()

        else:
            # skip other layer types (e.g., ActivationLayerConfig) as they don't need to be converted
            return layer_config



