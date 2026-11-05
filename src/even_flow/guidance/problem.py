# even_flow/guidance/problem.py  (library)
from typing import Protocol
import torch

class GuidanceProblem(Protocol):
    device: torch.device | str
    latent_shape: tuple[int, ...]

    def x1(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Flow model's clean-latent prediction (normalised latent space)."""

    def decode(self, mu_x: torch.Tensor) -> torch.Tensor:
        """Normalised latent -> decoded data (still normalised)."""

    def denorm_latent(self, z: torch.Tensor) -> torch.Tensor: ...
    
    def denorm_data(self, x: torch.Tensor) -> torch.Tensor: ...