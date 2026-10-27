import math

import torch

from mylm.model import GPT, GPTConfig

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
