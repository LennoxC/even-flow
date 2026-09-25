import even_flow
import torch
from abc import ABC, abstractmethod
from even_flow.config import TimeEncoderConfig, UNetConfig, ConvolutionalVariationalAutoencoderConfig, ResNetLayerConfig
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

    def forward(self, x):
        # Implement the forward pass for the UNet model with skip connections.
        skips = []
        for layer in self.encoder:
            x = layer(x)
            if hasattr(layer, 'emit_skip') and layer.emit_skip:
                skips.append(x)

        for layer in self.decoder:
            if hasattr(layer, 'receives_skip') and layer.receives_skip and skips:
                skip_connection = skips.pop()
                x = torch.cat((x, skip_connection), dim=1)  # Concatenate along the channel dimension
            x = layer(x)

        return x

    # UModelBase handles the _build_encoder and _build_decoder methods, along with other utility functions.
    # The _build_layer function is in the ModelBase class.

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

class FlowMatchingUNet(UNet):
    """
    flow matching UNet uses the same encoder/decoder structure, but has time encoding and concatenation in the latent space.
    therefore we need to add a time encoding (sinusoidal), a linear layer for projecting time to the latent,
    and a forward method that takes in time and concatenates it to the latent space before decoding.
    """
    def __init__(self, config: UNetConfig):
        super().__init__(config)
        if hasattr(self.config, 'time_encoding'):
            # make one time encoding per decoder layer. Use the input channels of the decoder layer as the channel dimension for the time encoding projection.
            convolutional_decoder_layers = [layer for layer in self.config.decoder_layers if hasattr(layer, 'in_channels')]
            self.time_encoders = torch.nn.ModuleList([self._make_time_encoding(self.config.time_encoding, layer_config.in_channels) for layer_config in convolutional_decoder_layers])
            # self.time_encoders = torch.nn.ModuleList([self._make_time_encoding(self.config.time_encoding, layer_config.in_channels) for layer_config in self.config.decoder_layers])

    def _make_time_encoding(self, time_encoding_config: TimeEncoderConfig, channel_dim: int = None):
        self.encoding_strategy = time_encoding_config.encoding_strategy
        self.time_encoding_dim = time_encoding_config.time_encoding_dim
        self.time_activation = time_encoding_config.activation

        encoding = getattr(even_flow.module.time_encoding, f"{self.encoding_strategy}TimeEncoding")(self.time_encoding_dim)
        projection = self._time_encoding_layer(self.time_encoding_dim, channel_dim)

        return torch.nn.Sequential(
            encoding,
            projection
        )

    def _reshape_t(self, t):
        """
        Reshape the time input to ensure it has the correct dimensions for processing.
        This method ensures that the time input is a 2D tensor of shape (B, 1), where B is the batch size.
        """
        if isinstance(t, (int, float)):
            t = torch.tensor([[t]], dtype=torch.float32)
        elif isinstance(t, torch.Tensor) and t.dim() == 1:
            t = t.view(-1, 1)  # Reshape to (B, 1)
        elif isinstance(t, torch.Tensor) and t.dim() == 0:
            t = t.view(1, 1)  # Reshape to (1, 1)
        return t


    def _time_encoding_layer(self, time_dim, channel_dim):
        """
        Create a linear layer to project the time encoding to the latent space.
        This layer will be used to transform the sinusoidal time encoding to match the latent space dimensions.

        Args:
            time_dim (int): The dimensionality of the time encoding.
            channel_dim (int): The number of channels in the latent space.
        """
        return torch.nn.Sequential(
            torch.nn.Linear(time_dim, channel_dim),
            getattr(torch.nn, self.time_activation)() if hasattr(torch.nn, self.time_activation) else torch.nn.GELU(),
            torch.nn.Linear(channel_dim, channel_dim)
        )

    def forward(self, x, t):
        # Encode the input
        skips = []
        for layer in self.encoder:
            x = layer(x)
            if hasattr(layer, 'emit_skip') and layer.emit_skip:
                skips.append(x)

        # Generate time encoding
        if not hasattr(self, 'time_encoders'):
            raise ValueError("Time encoders are not defined. Please ensure that the model is initialized with a valid TimeEncoderConfig.")

        t = self._reshape_t(t)

        for i, layer in enumerate(self.decoder):
            time_encoder = self.time_encoders[i]
            if hasattr(layer, 'receives_skip') and layer.receives_skip and skips:
                skip_connection = skips.pop()
                x = torch.cat((x, skip_connection), dim=1)  # Concatenate along the channel dimension
                x = x + time_encoder(t).view(x.size(0), x.size(1), 1, 1)  # Reshape time encoding to match the spatial dimensions of x
            else:
                x = x + time_encoder(t).view(x.size(0), -1, 1, 1)  # Reshape time encoding to match the spatial dimensions of x
            
            x = layer(x)

        return x
    
