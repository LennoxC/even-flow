"""
Observation masks for guided sampling.

Ground-truth masks (subsample a full field):   GridMask, RandomMask, UnionMask
Station observations (physical values given):  ManualMask

Everything returns a `MaskResult`, which can be unpacked into the legacy
5-tuple or passed straight to GaussNewtonSampler.inference(**result.inference_kwargs()).

Example
-------
    mask = GridMask(keep=10) | RandomMask(n_points=50, seed=0)
    res = mask.apply(guidance)                       # guidance: (C_all, H, W)
    out = sampler.inference(**res.inference_kwargs(), num_steps=15)
    print(res.n_obs)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
@dataclass
class MaskResult:
    guidance: np.ndarray            # (..., Cg, H, W) ground truth (or obs values), guidance channels only
    guidance_masked: np.ndarray     # (..., C_all, H, W) NaN wherever unobserved
    mask: np.ndarray                # (..., Cg, H, W) float32 {0, 1}, guidance channels only
    guidance_temporal: np.ndarray   # (..., C_all, H, W) full unmasked copy
    mask_temporal: np.ndarray       # (..., C_all, H, W) float32 {0, 1}, all channels
    n_guidance_channels: int = 3

    @property
    def n_obs(self) -> int:
        """Number of scalar observations the sampler will see."""
        return int(self.mask.sum())

    @property
    def batch_size(self) -> int:
        return self.mask.shape[0] if self.mask.ndim == 4 else 1

    @property
    def n_obs_per_element(self) -> List[int]:
        m = self.mask if self.mask.ndim == 4 else self.mask[None]
        return [int(x) for x in m.reshape(m.shape[0], -1).sum(1)]

    @classmethod
    def stack(cls, results: Sequence["MaskResult"]) -> "MaskResult":
        """Concatenate results along the batch axis. Elements may come from different mask types
        (and different guidance fields); each result may itself be unbatched (C,H,W) or batched (B,C,H,W)."""
        if len({r.n_guidance_channels for r in results}) != 1:
            raise ValueError("All results must use the same n_guidance_channels")
        if len({r.guidance_masked.shape[-3:] for r in results}) != 1:
            raise ValueError(
                "Results differ in (channels, H, W): " + str([r.guidance_masked.shape[-3:] for r in results])
                + ". Slice the guidance to the same channel count (e.g. guidance[:3]) before masking, "
                  "or build ManualMask with n_channels matching the others.")

        def cat(name):
            return np.concatenate([getattr(r, name) if getattr(r, name).ndim == 4 else getattr(r, name)[None]
                                   for r in results], axis=0)

        return cls(guidance=cat("guidance"), guidance_masked=cat("guidance_masked"), mask=cat("mask"),
                   guidance_temporal=cat("guidance_temporal"), mask_temporal=cat("mask_temporal"),
                   n_guidance_channels=results[0].n_guidance_channels)

    @classmethod
    def from_boolean_mask(cls, guidance, mask, n_guidance_channels: int = 3) -> "MaskResult":
        """guidance and mask share shape (..., C_all, H, W); mask is boolean."""
        guidance = np.asarray(guidance, dtype=np.float32)
        mask = np.asarray(mask, dtype=bool)
        if guidance.shape != mask.shape:
            raise ValueError(f"guidance {guidance.shape} and mask {mask.shape} must match")
        n = n_guidance_channels
        mask_f = mask.astype(np.float32)
        # np.where (not `== 0`) so genuinely observed zeros are kept as observations
        guidance_masked = np.where(mask, guidance, np.nan).astype(np.float32)
        return cls(
            guidance=guidance[..., :n, :, :],
            guidance_masked=guidance_masked,
            mask=mask_f[..., :n, :, :],
            guidance_temporal=guidance.copy(),
            mask_temporal=mask_f.copy(),
            n_guidance_channels=n,
        )

    def as_legacy_tuple(self):
        """(guidance, guidance_masked, mask, guidance_temporal, mask_temporal)."""
        return (self.guidance, self.guidance_masked, self.mask,
                self.guidance_temporal, self.mask_temporal)

    def inference_kwargs(self, device=None) -> Dict[str, object]:
        """Kwargs for GaussNewtonSampler.inference. Adds a batch dim to unbatched (C,H,W) data,
        so that `C = np.shape(mask)[1]` in the sampler is always the channel count.
 
        device=None: numpy arrays (samplers that call torch.as_tensor(..., device=...) move them themselves).
        device=...:  float32 torch tensors on that device (NaNs preserved in guidance_masked)."""
        g, m = self.guidance_masked, self.mask
        if m.ndim == 3:
            g, m = g[None], m[None]
        if device is not None:
            g = torch.as_tensor(g, dtype=torch.float32, device=device)
            m = torch.as_tensor(m, dtype=torch.float32, device=device)
        return {"guidance_masked": g, "mask": m}


# --------------------------------------------------------------------------- #
# Ground-truth masks
# --------------------------------------------------------------------------- #
class Mask(ABC):
    """Selects observed locations from a full (C, H, W) field.

    channels: restrict observations to these channel indices (None = all).
    seed:     makes stochastic masks reproducible (None = fresh randomness each call).
    """

    def __init__(self, channels: Optional[Sequence[int]] = None, seed: Optional[int] = None):
        self.channels = None if channels is None else tuple(channels)
        self.seed = seed

    @abstractmethod
    def _sample(self, shape: Tuple[int, int, int], rng: np.random.Generator) -> np.ndarray:
        """Return a boolean (C, H, W) array over all channels."""

    def generate(self, shape: Tuple[int, int, int], rng: Optional[np.random.Generator] = None) -> np.ndarray:
        if len(shape) != 3:
            raise ValueError(f"Expected (C, H, W) shape, got {shape}")
        if rng is None:
            rng = np.random.default_rng(self.seed)
        m = self._sample(tuple(shape), rng)
        if self.channels is not None:
            keep = np.zeros(shape[0], dtype=bool)
            keep[list(self.channels)] = True
            m = m & keep[:, None, None]
        return m

    def apply(self, guidance: np.ndarray, n_guidance_channels: int = 3,
              rng: Optional[np.random.Generator] = None) -> MaskResult:
        guidance = np.asarray(guidance)
        if guidance.ndim != 3:
            raise ValueError(f"guidance must be (C, H, W), got {guidance.shape}")
        m = self.generate(guidance.shape, rng)
        res = MaskResult.from_boolean_mask(guidance, m, n_guidance_channels)
        if res.n_obs == 0:
            raise ValueError("Mask selected no observations in the guidance channels.")
        return res

    def apply_batch(self, guidance: np.ndarray, batch_size: Optional[int] = None,
                    n_guidance_channels: int = 3, rng: Optional[np.random.Generator] = None) -> MaskResult:
        """Batched version of apply. guidance is (C,H,W) (shared across the batch, needs batch_size)
        or (B,C,H,W) (one field per element). Stochastic masks draw independently per element."""
        guidance = np.asarray(guidance)
        if guidance.ndim == 3:
            if batch_size is None:
                raise ValueError("batch_size is required when guidance is (C, H, W)")
            guidance = np.broadcast_to(guidance, (batch_size, *guidance.shape))
        elif guidance.ndim != 4 or (batch_size is not None and batch_size != guidance.shape[0]):
            raise ValueError(f"guidance {guidance.shape} incompatible with batch_size={batch_size}")
        if rng is None:
            rng = np.random.default_rng(self.seed)   # one generator, advanced across the batch
        return MaskResult.stack([self.apply(g, n_guidance_channels, rng) for g in guidance])

    __call__ = apply

    def __or__(self, other: "Mask") -> "UnionMask":
        return UnionMask(self, other)


class GridMask(Mask):
    """Regular grid: every `keep`-th pixel (int or (step_y, step_x)), starting at `offset` (y, x)."""

    def __init__(self, keep: Union[int, Tuple[int, int]], offset: Tuple[int, int] = (0, 0), **kw):
        super().__init__(**kw)
        self.step = (keep, keep) if isinstance(keep, int) else tuple(keep)
        self.offset = tuple(offset)
        if min(self.step) < 1:
            raise ValueError("keep must be >= 1")

    def _sample(self, shape, rng):
        m = np.zeros(shape, dtype=bool)
        (sy, sx), (oy, ox) = self.step, self.offset
        m[:, oy::sy, ox::sx] = True
        return m


class RandomMask(Mask):
    """Random pixels. Give exactly one of `n_points` (per channel) or `fraction` of the H*W pixels.

    shared_across_channels=False: each channel gets independent locations (matches the old build_random_mask).
    shared_across_channels=True:  all channels are observed at the same locations (like a real station).
    """

    def __init__(self, n_points: Optional[int] = None, fraction: Optional[float] = None,
                 shared_across_channels: bool = False, **kw):
        super().__init__(**kw)
        if (n_points is None) == (fraction is None):
            raise ValueError("Specify exactly one of n_points or fraction")
        self.n_points, self.fraction, self.shared = n_points, fraction, shared_across_channels

    def _sample(self, shape, rng):
        C, H, W = shape
        n = self.n_points if self.n_points is not None else int(round(self.fraction * H * W))
        if not 1 <= n <= H * W:
            raise ValueError(f"Cannot draw {n} points from {H * W} pixels")

        def draw():
            flat = np.zeros(H * W, dtype=bool)
            flat[rng.choice(H * W, size=n, replace=False)] = True
            return flat.reshape(H, W)

        if self.shared:
            return np.broadcast_to(draw(), shape).copy()
        return np.stack([draw() for _ in range(C)])


class UnionMask(Mask):
    """Observed wherever any sub-mask observes. Build with `a | b`."""

    def __init__(self, *masks: Mask, **kw):
        super().__init__(**kw)
        self.masks = masks

    def _sample(self, shape, rng):
        out = np.zeros(shape, dtype=bool)
        for m in self.masks:
            out |= m.generate(shape, rng)
        return out


# --------------------------------------------------------------------------- #
# Station observations
# --------------------------------------------------------------------------- #
@dataclass
class Observation:
    """One station. pos = (x, y) = (column, row). values: one per channel in physical units, None = unobserved."""
    pos: Tuple[int, int]
    values: Sequence[Optional[float]] = field(default_factory=list)

    @classmethod
    def from_tuple(cls, ob) -> "Observation":
        """Legacy form: (pos, temp_C, u, v[, mag])."""
        pos, *values = ob
        return cls(tuple(pos), list(values))


class ManualMask:
    """Builds guidance from explicit station observations.

    observations: one entry per batch element; each is an Observation / legacy tuple,
                  or a list of them for several stations in the same batch element.
    norm_params:  {channel_name: {"mean":..., "std":...}} used to normalise the values.
                  None = values are already normalised.
    channel_names: names aligned with channel index (only used with norm_params).
    temperature_in_celsius: channel 0 is given in Celsius and converted to Kelvin before normalising.
    """

    def __init__(self, observations: Sequence,
                 channel_names: Sequence[str], 
                 norm_params: Optional[Dict[str, Dict[str, float]]] = None,
                 shape: Tuple[int, int] = (1024, 1024),
                 n_channels: int = 3, temperature_in_celsius: bool = True):
        self.observations: List[List[Observation]] = [self._as_station_list(o) for o in observations]
        self.norm_params, self.channel_names = norm_params, tuple(channel_names)
        self.shape, self.n_channels, self.celsius = tuple(shape), n_channels, temperature_in_celsius
        if norm_params is not None:
            missing = [c for c in self.channel_names[:n_channels] if c not in norm_params]
            if missing:
                raise KeyError(f"norm_params missing channels: {missing}")

    @staticmethod
    def _as_station_list(o) -> List[Observation]:
        items = o if isinstance(o, list) else [o]
        return [x if isinstance(x, Observation) else Observation.from_tuple(x) for x in items]

    def build(self) -> MaskResult:
        H, W = self.shape
        B = len(self.observations)
        mask = np.zeros((B, self.n_channels, H, W), dtype=bool)
        guidance = np.zeros(mask.shape, dtype=np.float32)

        for b, stations in enumerate(self.observations):
            for ob in stations:
                x, y = ob.pos
                if not (0 <= x < W and 0 <= y < H):
                    raise ValueError(f"Position {ob.pos} outside grid {(W, H)} (pos is (x, y))")
                vals = (list(ob.values) + [None] * self.n_channels)[: self.n_channels]
                for c, v in enumerate(vals):
                    if v is None:
                        continue
                    if c == 0 and self.celsius:
                        v = v + 273.15
                    if self.norm_params is not None:
                        p = self.norm_params[self.channel_names[c]]
                        v = (v - p["mean"]) / p["std"]
                    guidance[b, c, y, x] = v
                    mask[b, c, y, x] = True

        res = MaskResult.from_boolean_mask(guidance, mask, self.n_channels)
        if res.n_obs == 0:
            raise ValueError("No observations were provided.")
        return res


def build_mask(guidance, keep):
    return GridMask(keep).apply(guidance).as_legacy_tuple()

def build_random_mask(guidance, keep):
    return RandomMask(n_points=keep).apply(guidance).as_legacy_tuple()