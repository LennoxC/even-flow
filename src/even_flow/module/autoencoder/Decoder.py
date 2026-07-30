import torch

class Decoder(torch.nn.Module):
    def __init__(self, layers: list[torch.nn.Module], receive_flags: list[bool]):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)
        self.receive_flags = receive_flags

    def forward(self, z, static_skips):
        # static_skips arrives fine->coarse (from StaticEncoder).
        # decoder runs coarse->fine, so consume in reverse order.
        skip_stack = list(reversed(static_skips))
        x = z
        for layer, receives in zip(self.layers, self.receive_flags):
            if receives:
                assert len(skip_stack) > 0, "Decoder layer expects a skip connection, but no more skips are available. You must provide static data in forward() if static_layers are configured."
                skip = skip_stack.pop(0)
                x = torch.cat([x, skip], dim=1)
            x = layer(x)
        return x