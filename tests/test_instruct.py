import torch

from mylm.instruct import (
    END,
    RESPONSE_MARK,
    build_prompt,
    parse_records,
    parse_request,
    requested_words,
    respond,
    split_records,
    word_coverage,
)
from mylm.model import GPT, GPTConfig
from mylm.tokenizer import BPETokenizer

SAMPLE = f"""Words: cat, sit
Features: Dialogue
Story:
The cat sat. "Hi," said Tom.
It was fun.
Summary: A cat meets Tom.
{END}
Summary: A dog runs.
Random sentence: The dog ran fast.
Story:
The dog ran fast and fell.
{END}
Summary: only a summary, no story
{END}
Story:
A story with no fields at all.
{END}
"""


def test_parse_records_handles_any_field_order_and_skips_storyless_entries():
    records = parse_records(SAMPLE)
    assert len(records) == 3  # the entry with a summary but no story is skipped
    first, second, third = records
    assert first.fields == {
        "Words": "cat, sit",
        "Features": "Dialogue",
        "Summary": "A cat meets Tom.",  # came after the story but still belongs to the request
    }
    assert first.story == 'The cat sat. "Hi," said Tom.\nIt was fun.'
    assert second.fields == {"Summary": "A dog runs.", "Random sentence": "The dog ran fast."}
    assert third.fields == {} and third.story == "A story with no fields at all."


def test_prompt_uses_a_fixed_field_order_and_ends_with_the_response_marker():
    prompt = build_prompt({"Summary": "S", "Words": "a, b"})
    assert prompt == (
        "### Instruction:\nWrite a short story.\nWords: a, b\nSummary: S\n\n" + RESPONSE_MARK
    )


def test_plain_text_requests_become_a_summary_and_labelled_ones_are_parsed():
    assert parse_request("a dragon who loves tea") == {"Summary": "a dragon who loves tea"}
    assert parse_request("Words: dog, ball\nFeatures: Dialogue") == {
        "Words": "dog, ball",
        "Features": "Dialogue",
    }


def test_word_coverage_counts_whole_words_ignoring_case():
    story = "The Cat sat in the category."
    assert word_coverage(["cat", "sat"], story) == 1.0
    assert word_coverage(["cat", "dog"], story) == 0.5
    assert word_coverage(["cate"], story) == 0.0  # "category" does not contain the word "cate"
    assert word_coverage([], story) == 1.0
    assert requested_words({"Words": "a, b ,c"}) == ["a", "b", "c"]


def test_split_records_holds_out_the_last_ones():
    train, held_out = split_records(list(range(10)), n_held_out=3)
    assert train == list(range(7)) and held_out == [7, 8, 9]


def test_generate_stops_when_the_stop_condition_is_met():
    cfg = GPTConfig(vocab_size=64, block_size=16, dim=32, n_layers=1, n_heads=4, dropout=0.0)
    model = GPT(cfg).eval()
    out = model.generate(
        torch.zeros((1, 2), dtype=torch.long), 10, stop=lambda ids: ids.shape[1] >= 5
    )
    assert out.shape == (1, 5)


def test_respond_returns_text_and_whether_the_story_finished():
    tokenizer = BPETokenizer()
    tokenizer.train("Write a short story. Words: cat. Summary: x. " * 20, 300)
    cfg = GPTConfig(vocab_size=tokenizer.vocab_size, block_size=128, dim=32, n_layers=1, n_heads=4)
    model = GPT(cfg).eval()
    story, finished = respond(model, tokenizer, {"Words": "cat"}, max_new_tokens=5, device="cpu")
    assert isinstance(story, str) and finished is False  # an untrained model never writes END


def test_an_entry_cut_off_before_its_end_marker_is_skipped():
    complete = f"Summary: S\nStory:\nIt ended.\n{END}\n"
    cut_off = "Summary: T\nStory:\nIt was cut off in the mi"
    assert [r.story for r in parse_records(complete + cut_off)] == ["It ended."]
    assert [r.story for r in parse_records(complete)] == ["It ended."]
