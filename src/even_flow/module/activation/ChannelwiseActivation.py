import torch
import torch.nn as nn
from typing import List, Union
from even_flow.module.activation.ActivationLayer import ActivationLayer

class ChannelwiseActivation(nn.Module):
    def __init__(self, activations: List[Union[str, nn.Module]], dim: int=1):
        super().__init__()
        self.dim = dim
        
        self.activations = nn.ModuleList()
        for act in activations:
            if isinstance(act, str):
                self.activations.append(ActivationLayer(act))
            elif isinstance(act, nn.Module):
                self.activations.append(act)
            else:
                raise ValueError(f"Invalid activation type: {type(act)}. Must be a string or a torch.nn.Module.")
        assert len(self.activations) > 0, "At least one activation function must be provided."
        
        assert dim in [1, 2, 3], "dim must be 1, 2, or 3"

    def forward(self, x):
        assert x.shape[self.dim] == len(self.activations), f"Number of channels in input ({x.shape[self.dim]}) must match number of activations ({len(self.activations)})"
        # Split the input tensor along the specified dimension and apply the corresponding activation function to each channel
        # assumes batch dim is channel 0

        if self.dim == 1:
            return torch.cat([self.activations[i](x[:, i:i+1, ...]) for i in range(x.shape[1])], dim=1)
        elif self.dim == 2:
            return torch.cat([self.activations[i](x[:, :, i:i+1, ...]) for i in range(x.shape[2])], dim=2)
        elif self.dim == 3:
            return torch.cat([self.activations[i](x[:, :, :, i:i+1, ...]) for i in range(x.shape[3])], dim=3)
        