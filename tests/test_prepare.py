import sys

import torch

from mylm.prepare import DATASETS, encode_in_blocks, main
from mylm.tokenizer import BPETokenizer


def test_dataset_registry_is_well_formed():
    assert {"shakespeare", "tinystories"} <= set(DATASETS)
    for dataset in DATASETS.values():
        assert dataset.url.startswith("https://")
        assert dataset.vocab_size > 256  # must leave room for merges beyond the 256 byte tokens


def test_prepare_writes_splits_and_tokenizer(tmp_path, monkeypatch):
    text = "Once upon a time there was a little cat. The cat liked to play. " * 200
    (tmp_path / "input.txt").write_text(text)  # already present, so nothing is downloaded
    argv = ["mylm-prepare", "--dataset", "tinystories", "--data-dir", str(tmp_path)]
    monkeypatch.setattr(sys, "argv", argv + ["--vocab-size", "300"])
    main()

    tokenizer = BPETokenizer.load(tmp_path / "tokenizer.json")
    train, val = torch.load(tmp_path / "train.pt"), torch.load(tmp_path / "val.pt")
    assert 256 < tokenizer.vocab_size <= 300  # may stop early once no pairs are left to merge
    assert len(train) > len(val) > 0
    assert tokenizer.decode(train.tolist()) + tokenizer.decode(val.tolist()) == text


def test_encoding_in_blocks_round_trips_and_matches_whole_text_encoding():
    text = "".join(f"Story {i}: the cat sat.\nThe dog ran home.\n" for i in range(60))
    tokenizer = BPETokenizer()
    tokenizer.train(text, 290)
    blocks = encode_in_blocks(tokenizer, text, block_chars=100)
    assert tokenizer.decode(blocks.tolist()) == text
    assert torch.equal(blocks, torch.tensor(tokenizer.encode(text)))  # cuts at newlines are exact
