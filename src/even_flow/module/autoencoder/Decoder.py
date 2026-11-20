import torch
import torch.nn.functional as F

class Decoder(torch.nn.Module):
    def __init__(self, layers: list[torch.nn.Module], receive_flags: list[bool], channelwise_activation: torch.nn.Module = None):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)
        self.receive_flags = receive_flags
        self.channelwise_activation = channelwise_activation

    def forward(self, z, static_skips):
        # static_skips arrives fine->coarse (from StaticEncoder).
        # decoder runs coarse->fine, so consume in reverse order.
        skip_stack = list(reversed(static_skips))
        x = z
        for layer, receives in zip(self.layers, self.receive_flags):
            if receives:
                assert len(skip_stack) > 0, "Decoder layer expects a skip connection, but no more skips are available. You must provide static data in forward() if static_layers are configured."
                skip = skip_stack.pop(0)
                
                if x.shape[-1:] != skip.shape[-1:]:
                    x = F.interpolate(
                        x,
                        size=skip.shape[-1:],
                        mode="bilinear",
                        align_corners=False,
                    )
                
                if x.shape[-2:] != skip.shape[-2:]:
                    x = F.interpolate(
                        x,
                        size=skip.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                
                x = torch.cat([x, skip], dim=1)
            x = layer(x)
        if self.channelwise_activation is not None:
            x = self.channelwise_activation(x)
        return x