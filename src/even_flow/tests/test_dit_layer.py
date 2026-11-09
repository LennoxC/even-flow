import pytest
import torch
from even_flow.module.dit.DiTLayer import DiTLayer, sincos_pos_embed

COND_DIM = 32

# (layer kwargs, spatial size)
CASES = {
    "1d_cond":     (dict(dim=1, channels=16, hidden_size=32, depth=2, num_heads=4, patch_size=4, cond_dim=COND_DIM), (32,)),
    "2d_cond":     (dict(dim=2, channels=32, hidden_size=64, depth=2, num_heads=4, patch_size=2, cond_dim=COND_DIM), (16, 16)),
    "2d_patch1":   (dict(dim=2, channels=16, hidden_size=32, depth=1, num_heads=2, patch_size=1, cond_dim=COND_DIM), (8, 8)),
    "2d_uncond":   (dict(dim=2, channels=32, hidden_size=64, depth=2, num_heads=4, patch_size=2, cond_dim=None), (16, 16)),
    "3d_cond":     (dict(dim=3, channels=16, hidden_size=32, depth=2, num_heads=4, patch_size=2, cond_dim=COND_DIM), (4, 8, 8)),
    "hidden_none": (dict(dim=2, channels=32, depth=1, num_heads=4, patch_size=2, cond_dim=COND_DIM), (8, 8)),  # hidden_size defaults to channels
}


@pytest.fixture(params=list(CASES), ids=list(CASES))
def case(request):
    kwargs, spatial = CASES[request.param]
    return kwargs, spatial


def make_inputs(kwargs, spatial, batch=2):
    x = torch.randn(batch, kwargs["channels"], *spatial)
    emb = torch.randn(batch, COND_DIM) if kwargs["cond_dim"] is not None else None
    return x, emb


@pytest.mark.fast
def test_forward_shape(case):
    kwargs, spatial = case
    layer = DiTLayer(**kwargs)
    x, emb = make_inputs(kwargs, spatial)
    assert layer(x, emb).shape == x.shape


@pytest.mark.fast
def test_zero_init_is_identity(case):
    kwargs, spatial = case
    layer = DiTLayer(**kwargs, zero_init=True).eval()
    x, emb = make_inputs(kwargs, spatial)
    assert torch.equal(layer(x, emb), x), "adaLN-Zero layer should be the identity at initialisation."


@pytest.mark.fast
def test_single_embedding_broadcasts_over_batch():
    kwargs, spatial = CASES["2d_cond"]
    layer = DiTLayer(**kwargs, zero_init=False)
    x, _ = make_inputs(kwargs, spatial, batch=3)
    emb = torch.randn(1, COND_DIM)  # e.g. a scalar t shared by the whole batch
    assert layer(x, emb).shape == x.shape


@pytest.mark.fast
def test_missing_embedding_raises():
    kwargs, spatial = CASES["2d_cond"]
    layer = DiTLayer(**kwargs)
    x, _ = make_inputs(kwargs, spatial)
    with pytest.raises(ValueError, match="time-conditioned"):
        layer(x)


@pytest.mark.fast
def test_unconditioned_layer_ignores_embedding():
    kwargs, spatial = CASES["2d_uncond"]
    layer = DiTLayer(**kwargs, zero_init=False).eval()
    x, _ = make_inputs(kwargs, spatial)
    assert torch.equal(layer(x), layer(x, torch.randn(2, COND_DIM)))


@pytest.mark.fast
def test_rejects_indivisible_spatial_size():
    kwargs, _ = CASES["2d_cond"]  # patch_size=2
    layer = DiTLayer(**kwargs)
    x = torch.randn(1, kwargs["channels"], 15, 16)
    with pytest.raises(ValueError, match="divisible"):
        layer(x, torch.randn(1, COND_DIM))


@pytest.mark.fast
def test_rejects_wrong_channels():
    kwargs, spatial = CASES["2d_cond"]
    layer = DiTLayer(**kwargs)
    with pytest.raises(ValueError, match="Channel mismatch"):
        layer(torch.randn(1, kwargs["channels"] + 1, *spatial), torch.randn(1, COND_DIM))


@pytest.mark.fast
def test_works_at_different_resolution():
    # fixed sin-cos positional embeddings should allow a resolution different from the training one
    kwargs, _ = CASES["2d_cond"]
    layer = DiTLayer(**kwargs, zero_init=False)
    for size in [(16, 16), (8, 24), (32, 32)]:
        x = torch.randn(1, kwargs["channels"], *size)
        assert layer(x, torch.randn(1, COND_DIM)).shape == x.shape


@pytest.mark.detailed
def test_output_depends_on_time_embedding():
    kwargs, spatial = CASES["2d_cond"]
    layer = DiTLayer(**kwargs, zero_init=False).eval()
    x, _ = make_inputs(kwargs, spatial)
    out_a = layer(x, torch.randn(2, COND_DIM))
    out_b = layer(x, torch.randn(2, COND_DIM))
    assert not torch.allclose(out_a, out_b), "Output does not change with the time embedding; conditioning is disconnected."


@pytest.mark.detailed
def test_determinism_in_eval():
    kwargs, spatial = CASES["2d_cond"]
    layer = DiTLayer(**kwargs, zero_init=False, dropout=0.5).eval()
    x, emb = make_inputs(kwargs, spatial)
    assert torch.equal(layer(x, emb), layer(x, emb))


@pytest.mark.detailed
@pytest.mark.parametrize("zero_init", [True, False], ids=["zero_init", "default_init"])
def test_gradients_flow(case, zero_init):
    kwargs, spatial = case
    layer = DiTLayer(**kwargs, zero_init=zero_init)
    x, emb = make_inputs(kwargs, spatial)
    target = torch.randn_like(x)
    optimizer = torch.optim.SGD(layer.parameters(), lr=1e-2)

    # adaLN-Zero needs several steps before every parameter sees gradient:
    #   step 1: only the output projection (everything upstream is multiplied by its zero weights)
    #   step 2: block gates / adaLN modulation, patch embedding
    #   step 3: attention and MLP internals (their gates are no longer zero)
    # We run 4 steps and check the gradients from the last backward pass.
    for _ in range(4):
        optimizer.zero_grad()
        loss = ((layer(x, emb) - target) ** 2).mean()  # not .mean() of the output: that is degenerate through LayerNorm
        loss.backward()
        optimizer.step()
        for name, p in layer.named_parameters():
            assert torch.isfinite(p).all(), f"{name} became non-finite during training"

    for name, p in layer.named_parameters():
        assert p.grad is not None, f"{name} got no gradient (disconnected from graph)"
        assert torch.isfinite(p.grad).all(), f"{name} has non-finite gradient"
        assert p.grad.abs().sum() > 0, f"{name} gradient is entirely zero"


@pytest.mark.fast
@pytest.mark.parametrize("hidden_size,grid", [(64, (8, 8)), (50, (4, 4, 4)), (32, (16,))])
def test_sincos_pos_embed(hidden_size, grid):
    emb = sincos_pos_embed(hidden_size, grid)
    n = 1
    for g in grid:
        n *= g
    assert emb.shape == (n, hidden_size)
    assert torch.isfinite(emb).all()
    assert len(torch.unique(emb, dim=0)) == n, "Every grid position should get a distinct embedding."