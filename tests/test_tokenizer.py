from mylm.tokenizer import BPETokenizer

CORPUS = "the cat sat on the mat. the cat ate 12 rats!\n" * 30


def trained(vocab_size=300):
    tok = BPETokenizer()
    tok.train(CORPUS, vocab_size)
    return tok


def test_round_trip_on_unseen_and_non_ascii_text():
    tok = trained()
    text = "the cat sat; Привіт 😀 _x 42\n\n  end"
    assert tok.decode(tok.encode(text)) == text


def test_training_compresses_text():
    tok = trained()
    assert len(tok.encode(CORPUS)) < len(CORPUS.encode()) / 2


def test_vocab_size_is_respected():
    assert trained(270).vocab_size == 270


def test_save_and_load_give_identical_encoding(tmp_path):
    tok = trained()
    tok.save(tmp_path / "tok.json")
    loaded = BPETokenizer.load(tmp_path / "tok.json")
    assert loaded.encode(CORPUS) == tok.encode(CORPUS)
