import math

import pytest
import torch

from mylm.model import GPT, GPTConfig, RMSNorm, Rotary, SwiGLU

CFG = GPTConfig(vocab_size=64, block_size=16, dim=32, n_layers=2, n_heads=4, dropout=0.0)


def test_output_shapes():
    model = GPT(CFG)
    idx = torch.randint(0, CFG.vocab_size, (3, 10))
    logits, loss = model(idx, idx)
    assert logits.shape == (3, 10, CFG.vocab_size)
    assert loss.ndim == 0


def test_initial_loss_is_close_to_uniform_guessing():
    torch.manual_seed(0)
    idx = torch.randint(0, CFG.vocab_size, (8, 16))
    targets = torch.randint(0, CFG.vocab_size, (8, 16))
    _, loss = GPT(CFG)(idx, targets)
    assert abs(loss.item() - math.log(CFG.vocab_size)) < 0.3


def test_model_cannot_see_the_future():
    torch.manual_seed(0)
    model = GPT(CFG).eval()
    a = torch.randint(0, CFG.vocab_size, (1, 12))
    b = a.clone()
    b[0, -1] = (b[0, -1] + 1) % CFG.vocab_size  # change only the last token
    logits_a, _ = model(a)
    logits_b, _ = model(b)
    assert torch.allclose(logits_a[:, :-1], logits_b[:, :-1], atol=1e-5)


def test_can_overfit_one_batch():
    torch.manual_seed(0)
    model = GPT(CFG)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    x = torch.randint(0, CFG.vocab_size, (4, 16))
    y = torch.roll(x, shifts=-1, dims=1)
    first = model(x, y)[1].item()
    for _ in range(100):
        loss = model(x, y)[1]
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first * 0.5


def test_generate_extends_the_prompt():
    model = GPT(CFG).eval()
    out = model.generate(torch.zeros((1, 3), dtype=torch.long), max_new_tokens=20, top_k=5)
    assert out.shape == (1, 23)  # longer than block_size is fine
    assert out.max() < CFG.vocab_size


def test_rmsnorm_outputs_unit_rms_and_keeps_direction():
    x = torch.randn(4, 7, 32) * 5 + 3  # large scale and a non-zero mean
    out = RMSNorm(32)(x)
    assert torch.allclose(out.pow(2).mean(-1), torch.ones(4, 7), atol=1e-4)
    # No mean subtraction: the output is just a rescaled copy of the input, so the mean survives.
    assert torch.allclose(out / x, (out / x)[..., :1].expand_as(x), atol=1e-4)


def test_rmsnorm_model_has_fewer_parameters_and_trains_the_same_way():
    layer = GPT(CFG)
    rms = GPT(GPTConfig(**{**CFG.to_dict(), "norm": "rmsnorm"}))
    n_norms = 2 * CFG.n_layers + 1
    assert layer.num_parameters() - rms.num_parameters() == n_norms * CFG.dim  # no bias vectors
    idx = torch.randint(0, CFG.vocab_size, (3, 10))
    logits, loss = rms(idx, idx)
    assert logits.shape == (3, 10, CFG.vocab_size) and loss.ndim == 0


def test_unknown_norm_is_rejected():
    with pytest.raises(ValueError, match="unknown norm"):
        GPT(GPTConfig(**{**CFG.to_dict(), "norm": "batchnorm"}))


def test_rope_preserves_vector_length():
    x = torch.randn(2, 4, 16, 8)
    assert torch.allclose(Rotary(8, 16)(x).norm(dim=-1), x.norm(dim=-1), atol=1e-5)


def test_rope_scores_depend_only_on_relative_position():
    torch.manual_seed(0)
    rotary = Rotary(head_dim=8, block_size=16)
    # The same q and k vector placed at every position, so only the rotation differs.
    q = rotary(torch.randn(1, 1, 1, 8).expand(1, 1, 16, 8))
    k = rotary(torch.randn(1, 1, 1, 8).expand(1, 1, 16, 8))
    scores = (q @ k.transpose(-2, -1))[0, 0]
    assert torch.allclose(scores[5, 2], scores[9, 6], atol=1e-5)  # both are 3 apart
    assert torch.allclose(scores[3, 3], scores[11, 11], atol=1e-5)  # both are 0 apart
    assert not torch.allclose(scores[5, 2], scores[5, 4], atol=1e-3)  # 3 apart vs 1 apart


def test_rope_model_drops_position_table_and_stays_causal():
    torch.manual_seed(0)
    rope = GPT(GPTConfig(**{**CFG.to_dict(), "pos": "rope"})).eval()
    assert rope.pos_emb is None
    assert GPT(CFG).num_parameters() - rope.num_parameters() == CFG.block_size * CFG.dim
    a = torch.randint(0, CFG.vocab_size, (1, 12))
    b = a.clone()
    b[0, -1] = (b[0, -1] + 1) % CFG.vocab_size
    assert torch.allclose(rope(a)[0][:, :-1], rope(b)[0][:, :-1], atol=1e-5)


def test_unknown_pos_is_rejected():
    with pytest.raises(ValueError, match="unknown pos"):
        GPT(GPTConfig(**{**CFG.to_dict(), "pos": "sinusoidal"}))


def test_swiglu_is_gated_by_the_up_projection():
    torch.manual_seed(0)
    ff = SwiGLU(dim=32, dropout=0.0)
    x = torch.randn(2, 5, 32)
    assert ff(x).shape == x.shape
    with torch.no_grad():
        ff.up.weight.zero_()  # the product with the gate path is zero, whatever the gate says
    assert torch.count_nonzero(ff(x)) == 0


def test_swiglu_keeps_parameter_count_close_to_plain_mlp():
    cfg = GPTConfig(vocab_size=64, block_size=16, dim=256, n_layers=2, n_heads=4)
    plain = GPT(cfg).num_parameters()
    gated = GPT(GPTConfig(**{**cfg.to_dict(), "ff": "swiglu"})).num_parameters()
    assert abs(gated - plain) / plain < 0.01


def test_unknown_ff_is_rejected():
    with pytest.raises(ValueError, match="unknown ff"):
        GPT(GPTConfig(**{**CFG.to_dict(), "ff": "relu"}))
