import torch

class StaticEncoder(torch.nn.Module):
    def __init__(self, layers: list[torch.nn.Module], emit_flags: list[bool]):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)
        self.emit_flags = emit_flags

    def forward(self, x):
        skips = []
        for layer, emit in zip(self.layers, self.emit_flags):
            x = layer(x)
            if emit:
                skips.append(x)
        return x, skips