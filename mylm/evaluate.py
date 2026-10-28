"""Measure how well a checkpoint follows "use these words" requests on held-out examples."""

import argparse
from pathlib import Path

import torch

from mylm.checkpoint import load_model
from mylm.instruct import parse_records, requested_words, respond, split_records, word_coverage
from mylm.train import pick_device


def evaluate_following(model, tokenizer, records, n=40, seed=0, **sampling):
    """Generate a story for the first `n` records that request words and score the results.

    word_coverage: average fraction of requested words used. all_words: share of stories that use
    every requested word. finished: share that ended on their own. reference: word_coverage of the
    human-written story, i.e. what a perfect answer scores.
    """
    items = [r for r in records if "Words" in r.fields][:n]
    torch.manual_seed(seed)
    covered, finished = [], []
    for record in items:
        story, done = respond(model, tokenizer, record.fields, **sampling)
        covered.append(word_coverage(requested_words(record.fields), story))
        finished.append(done)
    reference = [word_coverage(requested_words(r.fields), r.story) for r in items]
    mean = lambda xs: sum(xs) / max(1, len(xs))  # noqa: E731
    return {
        "n": len(items),
        "word_coverage": mean(covered),
        "all_words": mean([c == 1.0 for c in covered]),
        "finished": mean(finished),
        "reference": mean(reference),
    }


def format_scores(scores):
    return (
        f"{scores['n']} prompts | word coverage {scores['word_coverage']:.2f} "
        f"(human story: {scores['reference']:.2f}) | all words used {scores['all_words']:.0%} "
        f"| ended on its own {scores['finished']:.0%}"
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="checkpoints/sft.pt")
    p.add_argument("--data-dir", default="data/instruct")
    p.add_argument("-n", type=int, default=40, help="number of held-out prompts")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--max-new-tokens", type=int, default=450)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = args.device or pick_device()
    model, tokenizer = load_model(args.checkpoint, device)
    text = (Path(args.data_dir) / "input.txt").read_text(encoding="utf-8", errors="ignore")
    _, held_out = split_records(parse_records(text))
    scores = evaluate_following(
        model,
        tokenizer,
        held_out,
        args.n,
        temperature=args.temperature,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
    )
    print(format_scores(scores))


if __name__ == "__main__":
    main()
