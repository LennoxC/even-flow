import pytest
import even_flow.demos as demos
from even_flow.demos import *
from even_flow.module.autoencoder.VariationalAutoencoder import ConvolutionalVariationalAutoencoder
import torch

"""
Check all the models provided in the demos folder can be instantiated without errors.
"""

models = demos.__all__

@pytest.mark.parametrize("model", models)
@pytest.mark.demos
def test_model_instantiation(model):
    """
    Test that the model can be instantiated without errors.
    """
    try:
        model_config = globals()[model]
        model = ConvolutionalVariationalAutoencoder(model_config)

        # test forward pass
        x = torch.randn(1, *model_config.input_dim)
        if model_config.static_layers is not None:
            static = torch.randn(1, *model_config.static_dim)
            recon, mean, log_var = model(x, static=static)
        else:
            recon, mean, log_var = model(x)
        assert recon.shape == x.shape
    
    except Exception as e:
        pytest.fail(f"Model {model} failed to instantiate: {e}")
