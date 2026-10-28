"""Load a trained model and its tokenizer from a checkpoint written by the training scripts."""

import torch

from mylm.model import GPT, GPTConfig
from mylm.tokenizer import BPETokenizer


def load_model(path, device):
    """Returns (model in eval mode, tokenizer)."""
    ckpt = torch.load(path, map_location=device)
    model = GPT(GPTConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, BPETokenizer.from_merge_list(ckpt["tokenizer_merges"])
