import torch

from mylm.instruct import Record, build_prompt, build_response
from mylm.model import GPT, GPTConfig
from mylm.sft import IGNORE, build_examples, make_batch
from mylm.tokenizer import BPETokenizer

RECORDS = [
    Record({"Words": "cat, sit"}, "The cat sat down."),
    Record({"Summary": "A dog runs."}, "The dog ran fast and then it fell over in the mud."),
]


def tokenizer():
    tok = BPETokenizer()
    tok.train(" ".join(build_prompt(r.fields) + build_response(r.story) for r in RECORDS) * 5, 300)
    return tok


def test_examples_are_prompt_plus_response_and_long_ones_are_dropped():
    tok = tokenizer()
    examples = build_examples(RECORDS, tok, max_len=1000)
    ids, n_prompt = examples[0]
    assert tok.decode(ids[:n_prompt]) == build_prompt(RECORDS[0].fields)
    assert tok.decode(ids[n_prompt:]) == build_response(RECORDS[0].story)
    assert len(build_examples(RECORDS, tok, max_len=40)) < len(examples)


def test_loss_covers_only_response_tokens():
    tok = tokenizer()
    examples = build_examples(RECORDS, tok, max_len=1000)
    x, y = make_batch(examples)
    for row, (ids, n_prompt) in enumerate(examples):
        n = len(ids) - 1
        assert (y[row, : n_prompt - 1] == IGNORE).all()  # nothing is learned from the request
        assert y[row, n_prompt - 1 : n].tolist() == ids[n_prompt:]  # the whole response is
        assert (y[row, n:] == IGNORE).all()  # padding is skipped
        assert x[row, :n].tolist() == ids[:-1]


def test_fine_tuning_on_masked_batches_reduces_the_loss():
    torch.manual_seed(0)
    tok = tokenizer()
    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=128, dim=32, n_layers=2, n_heads=4)
    model = GPT(cfg)
    x, y = make_batch(build_examples(RECORDS, tok, max_len=127))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    first = model(x, y)[1].item()
    for _ in range(60):
        loss = model(x, y)[1]
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first * 0.5


def test_batch_width_is_rounded_up_to_a_multiple_but_capped():
    tok = tokenizer()
    examples = build_examples(RECORDS, tok, max_len=1000)
    x, y = make_batch(examples, multiple=64)
    assert x.shape[1] % 64 == 0 and x.shape == y.shape
    assert (y[:, -1] == IGNORE).all()  # the extra width is padding, so it is never trained on
    longest = max(len(ids) for ids, _ in examples) - 1
    capped = make_batch(examples, multiple=64, max_width=longest + 5)[0]
    assert capped.shape[1] == longest + 5  # rounding is cut short by the cap
    assert make_batch(examples, multiple=64, max_width=3)[0].shape[1] == longest  # never loses data
