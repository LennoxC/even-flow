import torch
from even_flow.config.AutoencoderConfig import AutoencoderConfig

class Autoencoder(torch.nn.Module):
    """
    A basic Autoencoder implementation.
    - Use AutoencoderConfig to configure the model.
    - ConvLayer is used for the encoder and decoder.
    - The encoder outputs the mean and log variance of the latent space.
    - This will be used by the flow matching module to sample from the latent space.
    """
    def __init__(self, config: AutoencoderConfig):
        super().__init__()
        self.config = config

        self.encoder = torch.nn.Sequential()

        self.decoder = torch.nn.Sequential()

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)