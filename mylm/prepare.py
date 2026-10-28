"""Download the dataset, train the tokenizer, and write tokenized train/validation splits."""

import argparse
import shutil
import urllib.request
from array import array
from dataclasses import dataclass
from pathlib import Path

import torch

from mylm.tokenizer import BPETokenizer

END_OF_TEXT = "<|endoftext|>"


@dataclass(frozen=True)
class Dataset:
    url: str
    vocab_size: int
    max_bytes: int | None = None  # download only the start of a large file
    val_fraction: float = 0.1
    tokenizer_chars: int | None = None  # learn BPE merges from at most this many leading characters


_HF = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main"
DATASETS = {
    # ~1.1 MB of Shakespeare plays, a standard small language-modelling corpus.
    "shakespeare": Dataset(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        512,
    ),
    # ~19 MB of short, simple stories written by an LLM for small models (Eldan & Li, 2023).
    # This is the dataset's validation file, which is plenty for a model of our size.
    "tinystories": Dataset(f"{_HF}/TinyStories-valid.txt", 2048),
    # The first 300 MB (~70M tokens) of the TinyStories training file, for larger models.
    "tinystories-large": Dataset(
        f"{_HF}/TinyStories-train.txt",
        4096,
        max_bytes=300_000_000,
        val_fraction=0.02,
        tokenizer_chars=30_000_000,
    ),
}


def download(url, path, max_bytes=None):
    """Save `url` to `path`; with `max_bytes`, fetch only the first max_bytes bytes."""
    if max_bytes is None:
        urllib.request.urlretrieve(url, path)
        return
    request = urllib.request.Request(url, headers={"Range": f"bytes=0-{max_bytes - 1}"})
    with urllib.request.urlopen(request) as response, open(path, "wb") as f:
        shutil.copyfileobj(response, f)


def encode_in_blocks(tokenizer, text, block_chars=64_000_000):
    """Encode a long text block by block (keeps memory low) into one tensor of token IDs.

    Blocks end just before a newline, so a word is never split across two blocks.
    """
    ids = array("q")
    pos = 0
    while pos < len(text):
        end = text.find("\n", pos + block_chars)
        end = len(text) if end < 0 else end
        ids.extend(tokenizer.encode(text[pos:end]))
        pos = end
    return (
        torch.frombuffer(ids, dtype=torch.int64) if len(ids) else torch.empty(0, dtype=torch.long)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, default="shakespeare")
    parser.add_argument("--data-dir", help="output folder (default: data, or data/<dataset>)")
    parser.add_argument("--vocab-size", type=int, help="default depends on the dataset")
    parser.add_argument("--val-fraction", type=float, help="default depends on the dataset")
    args = parser.parse_args()

    dataset = DATASETS[args.dataset]
    vocab_size = args.vocab_size or dataset.vocab_size
    val_fraction = args.val_fraction or dataset.val_fraction
    default_dir = "data" if args.dataset == "shakespeare" else f"data/{args.dataset}"
    out = Path(args.data_dir or default_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw = out / "input.txt"
    if not raw.exists():
        print(f"downloading {dataset.url}")
        download(dataset.url, raw, dataset.max_bytes)
    text = raw.read_text(encoding="utf-8", errors="ignore")
    if dataset.max_bytes is not None and END_OF_TEXT in text:  # drop the story cut off by the range
        text = text[: text.rindex(END_OF_TEXT) + len(END_OF_TEXT)] + "\n"
    print(f"{len(text):,} characters")

    # The tokenizer only ever sees the training portion, so validation stays unseen.
    split = int(len(text) * (1 - val_fraction))
    tokenizer = BPETokenizer()
    tokenizer.train(text[: min(split, dataset.tokenizer_chars or split)], vocab_size)
    tokenizer.save(out / "tokenizer.json")

    train_ids = encode_in_blocks(tokenizer, text[:split])
    val_ids = encode_in_blocks(tokenizer, text[split:])
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
