import sys

import torch

from mylm.prepare import DATASETS, main
from mylm.tokenizer import BPETokenizer


def test_dataset_registry_is_well_formed():
    assert {"shakespeare", "tinystories"} <= set(DATASETS)
    for url, vocab_size in DATASETS.values():
        assert url.startswith("https://")
        assert vocab_size > 256  # must leave room for merges beyond the 256 byte tokens


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
