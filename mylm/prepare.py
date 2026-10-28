"""Download the dataset, train the tokenizer, and write tokenized train/validation splits."""

import argparse
import urllib.request
from pathlib import Path

import torch

from mylm.tokenizer import BPETokenizer

# name -> (download URL, default vocabulary size)
DATASETS = {
    # ~1.1 MB of Shakespeare plays, a standard small language-modelling corpus.
    "shakespeare": (
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        512,
    ),
    # ~19 MB of short, simple stories written by an LLM for small models (Eldan & Li, 2023).
    # This is the dataset's validation file, which is plenty for a model of our size.
    "tinystories": (
        "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt",
        2048,
    ),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, default="shakespeare")
    parser.add_argument("--data-dir", help="output folder (default: data, or data/<dataset>)")
    parser.add_argument("--vocab-size", type=int, help="default depends on the dataset")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    args = parser.parse_args()

    url, default_vocab = DATASETS[args.dataset]
    vocab_size = args.vocab_size or default_vocab
    default_dir = "data" if args.dataset == "shakespeare" else f"data/{args.dataset}"
    out = Path(args.data_dir or default_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw = out / "input.txt"
    if not raw.exists():
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, raw)
    text = raw.read_text(encoding="utf-8")
    print(f"{len(text):,} characters")

    # The tokenizer only ever sees the training portion, so validation stays unseen.
    split = int(len(text) * (1 - args.val_fraction))
    tokenizer = BPETokenizer()
    tokenizer.train(text[:split], vocab_size)
    tokenizer.save(out / "tokenizer.json")

    train_ids = torch.tensor(tokenizer.encode(text[:split]), dtype=torch.long)
    val_ids = torch.tensor(tokenizer.encode(text[split:]), dtype=torch.long)
    torch.save(train_ids, out / "train.pt")
    torch.save(val_ids, out / "val.pt")
    print(
        f"vocab {tokenizer.vocab_size} | train {len(train_ids):,} tokens "
        f"| val {len(val_ids):,} tokens"
    )
    print(f"compression: {split / len(train_ids):.2f} characters per token")
    print(f"wrote {out}/tokenizer.json, train.pt, val.pt")


if __name__ == "__main__":
    main()
