"""Generate text from a trained checkpoint."""

import argparse

import torch

from mylm.model import GPT, GPTConfig
from mylm.tokenizer import BPETokenizer
from mylm.train import pick_device


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="checkpoints/model.pt")
    p.add_argument("--prompt", default="\n")
    p.add_argument("--max-new-tokens", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--no-cache", action="store_true", help="recompute everything at every step")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
    device = args.device or pick_device()
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = GPT(GPTConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tokenizer = BPETokenizer.from_merge_list(ckpt["tokenizer_merges"])

    idx = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)
    out = model.generate(
        idx, args.max_new_tokens, args.temperature, args.top_k, use_cache=not args.no_cache
    )
    print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
