import torch
from even_flow.config import TimeEncoderConfig
import even_flow.module.time_encoding

class TimeEmbedding(torch.nn.Module):
    def __init__(self, cfg: TimeEncoderConfig):
        super().__init__()
        dim = cfg.time_encoding_dim
        encoding_cls = getattr(even_flow.module.time_encoding, f"{cfg.encoding_strategy}TimeEncoding")
        activation_cls = getattr(torch.nn, cfg.activation)

        # optional: TimeEncoderConfig.time_scale (default 1.0 = legacy behaviour)
        time_scale = getattr(cfg, "time_scale", 1.0)
        kwargs = {"time_scale": time_scale} if time_scale != 1.0 else {}
        self.encoding = encoding_cls(dim, **kwargs)

        self.mlp = torch.nn.Sequential(torch.nn.Linear(dim, dim), activation_cls(), torch.nn.Linear(dim, dim))
        self.out_dim = dim

    def forward(self, t: torch.Tensor):
        t = t.reshape(-1, 1) # accepts scalar, (B,), or (B, 1)
        return self.mlp(self.encoding(t))


class TimeInjection(torch.nn.Module):
    def __init__(self, emb_dim: int, channels: int):
        super().__init__()
        self.channels = channels
        self.proj = torch.nn.Sequential(torch.nn.SiLU(), torch.nn.Linear(emb_dim, channels))

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor):
        return x + self.proj(t_emb)[:, :, None, None]