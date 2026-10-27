"""Byte-level BPE tokenizer: learn merges from text, then encode and decode.

Text is first split into word-like chunks (a word with its leading space, a run of digits,
a run of punctuation, ...). Merges never cross chunk boundaries, which keeps training fast
and stops tokens like "dog." from swallowing the space after them.
"""

import json
import re
from collections import Counter
from itertools import pairwise
from pathlib import Path

# Every character is matched by exactly one alternative, so joining the chunks restores the text.
_CHUNK = re.compile(r"\s?[^\W\d_]+|\s?\d+|\s?[^\s\w]+|\s?_+|\s+")


def count_pairs(ids):
    """Count how often each adjacent pair appears in a list of token IDs."""
    return Counter(pairwise(ids))


def merge(ids, pair, new_id):
    """Replace every occurrence of `pair` in `ids` with `new_id`."""
    out = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    def __init__(self, merges=None):
        self.merges = {}  # (id_a, id_b) -> new_id, in the order they were learned
        self._cache = {}  # chunk -> token IDs
        self._vocab = {i: bytes([i]) for i in range(256)}
        for pair, new_id in (merges or {}).items():
            self._add_merge(pair, new_id)

    @property
    def vocab_size(self):
        return 256 + len(self.merges)

    def _add_merge(self, pair, new_id):
        self.merges[pair] = new_id
        self._vocab[new_id] = self._vocab[pair[0]] + self._vocab[pair[1]]

    def train(self, text, vocab_size):
        """Learn merges until the vocabulary has `vocab_size` entries (256 byte tokens + merges)."""
        chunk_counts = Counter(_CHUNK.findall(text))
        words = {chunk: list(chunk.encode("utf-8")) for chunk in chunk_counts}
        for new_id in range(self.vocab_size, vocab_size):
            pairs = Counter()
            for chunk, ids in words.items():
                for pair in pairwise(ids):
                    pairs[pair] += chunk_counts[chunk]
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            words = {chunk: merge(ids, best, new_id) for chunk, ids in words.items()}
            self._add_merge(best, new_id)
        self._cache.clear()

    def _encode_chunk(self, chunk):
        if chunk in self._cache:
            return self._cache[chunk]
        ids = list(chunk.encode("utf-8"))
        while len(ids) > 1:
            # Apply the earliest-learned merge available, exactly as training did.
            best = min(count_pairs(ids), key=lambda p: self.merges.get(p, float("inf")))
            if best not in self.merges:
                break
            ids = merge(ids, best, self.merges[best])
        self._cache[chunk] = ids
        return ids

    def encode(self, text):
        """Text -> list of token IDs."""
        return [i for chunk in _CHUNK.findall(text) for i in self._encode_chunk(chunk)]

    def decode(self, ids):
        """List of token IDs -> text."""
        return b"".join(self._vocab[i] for i in ids).decode("utf-8", errors="replace")

    def save(self, path):
        data = {"merges": [[a, b] for (a, b) in self.merges]}
        Path(path).write_text(json.dumps(data))

    @classmethod
    def load(cls, path):
        return cls.from_merge_list(json.loads(Path(path).read_text())["merges"])

    @classmethod
    def from_merge_list(cls, merge_list):
        tok = cls()
        for a, b in merge_list:
            tok._add_merge((a, b), tok.vocab_size)
        return tok
