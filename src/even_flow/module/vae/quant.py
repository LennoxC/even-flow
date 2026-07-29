import torch

class VAELatentEncoder(torch.nn.Module):
    """
    Projects encoder features into VAE latent moments (mean, log-variance).
    """
    def __init__(self, channels, latent_dim, dim=2, logvar_clamp=(-30.0, 20.0)):
        super().__init__()
        self.channels = channels
        self.latent_dim = latent_dim
        self.logvar_clamp = logvar_clamp

        ConvNd = getattr(torch.nn, f"Conv{dim}d")
        self.to_moments = ConvNd(channels, latent_dim * 2, kernel_size=1)

    def forward(self, x):
        moments = self.to_moments(x)
        mean, log_var = torch.chunk(moments, 2, dim=1)
        log_var = torch.clamp(log_var, *self.logvar_clamp)
        return mean, log_var

class PostQuantConv(torch.nn.Module):
    """
    Projects a sampled (or mean, at inference) latent back to the decoder's
    working channel count.
    """
    def __init__(self, latent_dim, channels, dim=2):
        super().__init__()
        self.latent_dim = latent_dim
        self.channels = channels

        ConvNd = getattr(torch.nn, f"Conv{dim}d")
        self.from_latent = ConvNd(latent_dim, channels, kernel_size=1)

    def forward(self, z):
        return self.from_latent(z)

def reparameterize(mean, log_var):
    """
    Sample z ~ N(mean, exp(log_var)) via the reparameterization trick.
    """
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return mean + eps * std

"""
Using in a VAE forward pass:

mean, log_var = self.encoder_head(h)          # VAELatentEncoder
z = reparameterize(mean, log_var)             # sample during training
# z = mean                                    # use this at inference for deterministic decode
h = self.post_quant_conv(z)                   # PostQuantConv
recon = self.decoder_body(h)
"""

"""
Computing KL Loss:
kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=1).mean()
"""