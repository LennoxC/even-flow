import even_flow
from even_flow.module.time_encoding.TimeEmbedding import TimeEmbedding, TimeInjection
import torch
import torch.nn.functional as F
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


class StaticInject(torch.nn.Module):
    """
    Adds a zero-initialised 1x1-projected copy of fixed fields (topography, land-sea mask,
    lat/lon, ...) to a feature map. Convs with reflect padding are translation-equivariant and
    carry no absolute position, so without this the flow model cannot know where the terrain is.

    static: (B or 1, C_static, H, W) at any resolution; it is area-averaged to the feature map size.
    """
    def __init__(self, static_channels: int, channels: int):
        super().__init__()
        self.proj = torch.nn.Conv2d(static_channels, channels, kernel_size=1)
        torch.nn.init.zeros_(self.proj.weight)
        torch.nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        if static.shape[-2:] != x.shape[-2:]:
            static = F.adaptive_avg_pool2d(static, x.shape[-2:])
        return x + self.proj(static.to(x.dtype))


class UNetBase(UModelBase):
    def __init__(self, config: UNetConfig):
        super().__init__()
        self.config = config
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()

    def _inject_time(self, x, emb, layer_index):
        return x # plain unet does not inject time

    def forward(self, x):
        x, skips = self._run_encoder(x)
        return self._run_decoder(x, skips)

    # UModelBase handles the _build_encoder and _build_decoder methods, along with other utility functions.
    # The _build_layer function is in the ModelBase class.

    def decode(self, z):
        return self.decoder(z)

    def _apply_layer(self, layer, x, emb):
        if getattr(layer, "takes_time_emb", False):
            return layer(x, emb)
        return layer(x)

    def _run_encoder(self, x, emb=None, static=None):
        skips = []
        injections = getattr(self, "static_injections", None)
        for i, layer in enumerate(self.encoder):
            x = self._apply_layer(layer, x, emb)
            if hasattr(layer, 'emit_skip') and layer.emit_skip:
                # inject fixed fields BEFORE the skip is stored, so both the skip and the
                # continuing path carry them.
                if static is not None and injections is not None and str(i) in injections:
                    x = injections[str(i)](x, static)
                skips.append(x)
        return x, skips

    def _run_decoder(self, x, skips, emb=None):
        for i, layer in enumerate(self.decoder):
            if getattr(layer, "receives_skip", False):
                x = torch.cat((x, skips.pop()), dim=1)
            x = self._inject_time(x, emb, i)
            x = self._apply_layer(layer, x, emb)
        return x

    def encode(self, x):
        return self._run_encoder(x)[0]   # was self.encoder(x), which can't pass emb


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

    NEW (opt-in, defaults keep the old state_dict):
    - Decoder layers that condition on time themselves (takes_time_emb, e.g. ResNetLayer with cond_dim)
      do not also get an additive TimeInjection.
    - config.static_channels > 0 adds a StaticInject after every skip-emitting encoder layer;
      forward(x, t, static) must then be given the static fields.
    """
    def __init__(self, config: UNetConfig):
        super().__init__(config)
        self.time_embedding = TimeEmbedding(config.time_encoding) if hasattr(config, 'time_encoding') else None

        self.time_injections = torch.nn.ModuleDict({
            str(i): TimeInjection(self.time_embedding.out_dim, layer.in_channels)
            for i, layer in enumerate(self.decoder)
            if hasattr(layer, 'in_channels') and not getattr(layer, 'takes_time_emb', False)
        })

        self.static_channels = getattr(config, "static_channels", 0) or 0
        self.static_injections = torch.nn.ModuleDict({
            str(i): StaticInject(self.static_channels, layer.out_channels)
            for i, layer in enumerate(self.encoder)
            if self.static_channels > 0 and getattr(layer, "emit_skip", False) and hasattr(layer, "out_channels")
        })

    def _inject_time(self, x, emb, layer_index):
        key = str(layer_index)
        if key not in self.time_injections:
            return x
        inj = self.time_injections[key]
        if x.size(1) != inj.channels:
            raise ValueError(f"Channel mismatch: x has {x.size(1)} channels, but TimeInjection expects {inj.channels} channels.")
        return inj(x, emb)

    def forward(self, x, t, static=None):
        if len(self.static_injections) > 0 and static is None:
            raise ValueError("This model was built with static_channels > 0: pass static=(B or 1, C_static, H, W).")
        emb = self.time_embedding(t)
        x, skips = self._run_encoder(x, emb, static)
        return self._run_decoder(x, skips, emb)

    def encode(self, x, t=None):
        # so calculate_latent_dimensionality() etc. still work with a conditioned DiT in the encoder
        if t is None:
            t = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
        return self._run_encoder(x, self.time_embedding(t))[0]