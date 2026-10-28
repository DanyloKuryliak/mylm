"""Train the model on the tokenized dataset produced by `mylm-prepare`."""

import argparse
import math
import time
from pathlib import Path

import torch

from mylm.data import get_batch
from mylm.model import GPT, GPTConfig
from mylm.tokenizer import BPETokenizer


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def lr_at(step, max_steps, max_lr, warmup):
    """Linear warmup, then cosine decay to 10% of max_lr."""
    if step < warmup:
        return max_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, max_steps - warmup)
    return max_lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


@torch.no_grad()
def estimate_loss(model, data, args, device):
    """Average loss over several random batches (less noisy than a single batch)."""
    model.eval()
    losses = [
        model(*get_batch(data, args.batch_size, args.block_size, device))[1].item()
        for _ in range(args.eval_batches)
    ]
    model.train()
    return sum(losses) / len(losses)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out", default="checkpoints/model.pt")
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--norm", choices=["layernorm", "rmsnorm"], default="layernorm")
    p.add_argument("--pos", choices=["learned", "rope"], default="learned")
    p.add_argument("--ff", choices=["gelu", "swiglu"], default="gelu")
    p.add_argument("--kv-heads", type=int, default=None, help="key/value heads (default: --heads)")
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--eval-interval", type=int, default=500)
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default=None, help="cuda, mps or cpu (default: auto)")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = args.device or pick_device()
    data_dir = Path(args.data_dir)
    train_data = torch.load(data_dir / "train.pt")
    val_data = torch.load(data_dir / "val.pt")
    tokenizer = BPETokenizer.load(data_dir / "tokenizer.json")

    cfg = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        dim=args.dim,
        n_layers=args.layers,
        n_heads=args.heads,
        dropout=args.dropout,
        norm=args.norm,
        pos=args.pos,
        ff=args.ff,
        n_kv_heads=args.kv_heads,
    )
    model = GPT(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    print(
        f"device {device} | {model.num_parameters():,} parameters | "
        f"{len(train_data):,} train tokens | vocab {cfg.vocab_size}"
    )

    start = time.time()
    for step in range(args.steps + 1):
        if step % args.eval_interval == 0 or step == args.steps:
            train_loss = estimate_loss(model, train_data, args, device)
            val_loss = estimate_loss(model, val_data, args, device)
            print(
                f"step {step:5d} | train loss {train_loss:.3f} | val loss {val_loss:.3f} "
                f"| {time.time() - start:6.1f}s"
            )
        if step == args.steps:
            break

        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, args.steps, args.lr, args.warmup)
        x, y = get_batch(train_data, args.batch_size, args.block_size, device)
        _, loss = model(x, y)  # forward pass: predictions and loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()  # backpropagation: gradient of loss per weight
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()  # nudge the weights downhill

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": cfg.to_dict(),
            "model": model.state_dict(),
            "tokenizer_merges": [list(pair) for pair in tokenizer.merges],
        },
        out,
    )
    print(f"saved {out}")

    prompt = torch.zeros((1, 1), dtype=torch.long, device=device)
    sample = model.generate(prompt, max_new_tokens=200, top_k=40)[0].tolist()
    print("\n--- sample ---\n" + tokenizer.decode(sample))


if __name__ == "__main__":
    main()
