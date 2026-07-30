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