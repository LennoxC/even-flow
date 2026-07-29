import torch

class ActivationLayer(torch.nn.Module):
    """
    A simple activation layer that applies a specified activation function to the input tensor.
    This exists so that activation functions can be specified in the model config and used in the model:
        - Some layers (like ConvLayer) don't have an activation function, so this layer can be used to add one.
    """
    def __init__(self, activation: str):
        super().__init__()
        self.activation = getattr(torch.nn, activation)()  # Dynamically get the activation function from torch.nn

    def forward(self, x):
        return self.activation(x)