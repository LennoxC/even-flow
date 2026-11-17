from __future__ import annotations
from dataclasses import dataclass

@dataclass
class ActivationLayerConfig:
    """
    Activation layers can be placed in encoder/decoder/static networks to introduce non-linearities. 
    These are often used following layers that do not have a built-in activation function, such as convolutional layers or attention layers.

    Attributes:
        activation: str - the activation function to use. This can match the class name of any pytorch activation function (e.g. "ReLU", "GELU", "LeakyReLU", etc.). The activation function will be instantiated with default parameters.
    """
    activation: str = "GELU" # activation function

@dataclass
class ChannelwiseActivationLayerConfig:
    """
    Uses activation functions to bound variables: intended to be used as the final layer of a VAE decoder to bound the output to a specific range.
    E.g. for rainfall prediction, a ReLU activation function can ensure the output is non-negative.

    Attributes:
        activations: list[str] - the activation functions to use for each channel. This can match the class name of any pytorch activation function (e.g. "ReLU", "GELU", "LeakyReLU", etc.). The activation functions will be instantiated with default parameters.
        dim: int - the dimension along which to apply the activations. This should match the channel dimension of the input tensor (e.g. 1 for (batch, channels, height, width), 2 for (batch, height, channels, width), 3 for (batch, height, width, channels)).
    """
    activations: list[str] = None # list of activation functions
    dim: int = 1 # dimension along which to apply the activations