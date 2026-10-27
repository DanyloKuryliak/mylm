import torch

from mylm.data import get_batch


def test_targets_are_inputs_shifted_by_one():
    data = torch.arange(1000)
    x, y = get_batch(data, batch_size=4, block_size=16)
    assert x.shape == y.shape == (4, 16)
    assert torch.equal(x[:, 1:], y[:, :-1])
    assert torch.equal(y[:, -1], x[:, -1] + 1)  # on arange data, the next token is +1
