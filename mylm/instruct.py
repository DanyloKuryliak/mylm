"""The instruction format shared by fine-tuning, evaluation and chat.

A request is a few labelled fields (words to use, features, a sentence to include, a summary)
rendered into a fixed prompt; the model answers with a story. The fields mirror the labels of the
TinyStories-Instruct dataset, which is where the training examples come from.
"""

import re
from dataclasses import dataclass

import torch

END = "<|endoftext|>"
RESPONSE_MARK = "### Response:\n"
LABELS = ("Words", "Features", "Random sentence", "Summary")  # order used inside a prompt
_LABEL = re.compile(r"^(Words|Features|Random sentence|Summary|Story):[ \t]*", re.M)


@dataclass
class Record:
    fields: dict  # label -> text, for the labels in LABELS that the example has
    story: str


def parse_fields(chunk):
    """Split text into {label: text}; a field runs until the next label at the start of a line."""
    matches = list(_LABEL.finditer(chunk))
    sections = {}
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(chunk)
        sections.setdefault(m.group(1), chunk[m.end() : end].strip())
    return sections


def parse_records(text):
    """TinyStories-Instruct text -> list of Record. Entries without a story are skipped.

    Field order varies between entries and a field such as the summary may come after the story.
    """
    records = []
    for chunk in text.split(END):
        sections = parse_fields(chunk)
        story = sections.pop("Story", "")
        if story:
            records.append(Record(sections, story))
    return records


def parse_request(text):
    """What a user typed -> fields. Plain text is treated as a summary of the wanted story."""
    fields = parse_fields(text)
    fields.pop("Story", None)
    return fields or {"Summary": text.strip()}


def build_prompt(fields):
    lines = ["### Instruction:", "Write a short story."]
    lines += [f"{label}: {fields[label]}" for label in LABELS if label in fields]
    return "\n".join(lines) + "\n\n" + RESPONSE_MARK


def build_response(story):
    return story.strip() + "\n" + END


def requested_words(fields):
    return [w.strip() for w in fields.get("Words", "").split(",") if w.strip()]


def word_coverage(words, story):
    """Fraction of the requested words that appear in the story as whole words."""
    if not words:
        return 1.0
    found = sum(bool(re.search(rf"\b{re.escape(w)}\b", story, re.I)) for w in words)
    return found / len(words)


@torch.no_grad()
def respond(model, tokenizer, fields, max_new_tokens=450, temperature=0.8, top_k=40, device=None):
    """Generate a story for `fields`. Returns (story, finished); finished = it ended by itself."""
    device = device or next(model.parameters()).device
    prompt_ids = tokenizer.encode(build_prompt(fields))
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    def finished(ids):
        return END in tokenizer.decode(ids[0, len(prompt_ids) :].tolist())

    out = model.generate(idx, max_new_tokens, temperature, top_k, stop=finished)
    text = tokenizer.decode(out[0, len(prompt_ids) :].tolist())
    return text.split(END)[0].strip(), END in text


def split_records(records, n_held_out=500):
    """(training records, held-out records): the last `n_held_out` are never trained on."""
    return records[:-n_held_out], records[-n_held_out:]
