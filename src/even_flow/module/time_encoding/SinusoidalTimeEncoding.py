import torch


class SinusoidalTimeEncoding(torch.nn.Module):
    """
    Sinusoidal embedding of a scalar time.

    time_scale multiplies t before encoding. The original code fed raw t in [0, 1] to frequencies
    in [1e-4, 1] rad per unit, so t=0.95 and t=0.99 differ by ~0.04 rad in the fastest channel,
    which makes noise levels near t=1 (where fine detail is decided) hard to tell apart.
    DiT/ADM-style models use t*1000. Use time_scale=1000 for NEW flow models.

    Default 1.0 reproduces the old encoding exactly: it is NOT stored in the state_dict, so a
    checkpoint trained with time_scale=1 must be loaded with time_scale=1, and vice versa.
    """
    def __init__(self, dim, time_scale: float = 1.0):
        super().__init__()
        self.dim = dim
        self.time_scale = float(time_scale)
        half_dim = dim // 2
        frequencies = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -(torch.log(torch.tensor(10000.0)) / (half_dim - 1)))
        self.register_buffer("frequencies", frequencies)

    def forward(self, t):
        t = t.reshape(-1, 1).to(self.frequencies.dtype) * self.time_scale
        emb = t * self.frequencies.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(t)], dim=1)

        return emb