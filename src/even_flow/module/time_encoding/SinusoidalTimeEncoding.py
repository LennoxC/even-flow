import torch

class SinusoidalTimeEncoding(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        half_dim = dim // 2
        frequencies = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -(torch.log(torch.tensor(10000.0)) / (half_dim - 1)))
        self.register_buffer("frequencies", frequencies)

    def forward(self, t):
        t = t.view(-1, 1)
        emb = t * self.frequencies.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(t)], dim=1)

        return emb