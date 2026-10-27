"""Batching: turn one long tensor of token IDs into (input, target) training pairs."""

import torch


def get_batch(data, batch_size, block_size, device="cpu"):
    """Sample random windows from a 1-D tensor of token IDs.

    x: (batch_size, block_size) inputs
    y: (batch_size, block_size) targets: x shifted one token to the right
    """
    starts = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[s : s + block_size] for s in starts])
    y = torch.stack([data[s + 1 : s + 1 + block_size] for s in starts])
    return x.to(device), y.to(device)
