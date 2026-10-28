"""Supervised fine-tuning: teach a pretrained model to answer requests with stories.

Each training example is a prompt (the request) followed by the response (the story). The loss is
computed only on the response tokens, so the model learns to answer, not to write requests.
"""

import argparse
import time
from pathlib import Path

import torch

from mylm.checkpoint import load_model
from mylm.evaluate import evaluate_following, format_scores
from mylm.instruct import build_prompt, build_response, parse_records, respond, split_records
from mylm.prepare import download
from mylm.train import lr_at, pick_device

INSTRUCT_URL = (
    "https://huggingface.co/datasets/roneneldan/TinyStoriesInstruct/resolve/main/"
    "TinyStories-Instruct-valid.txt"
)
IGNORE = -100  # target value that the loss skips (cross_entropy's ignore_index)
PAD_MULTIPLE = 64


def build_examples(records, tokenizer, max_len):
    """Records -> [(prompt + response ids, prompt length)]; examples over max_len are dropped."""
    examples = []
    for record in records:
        prompt = tokenizer.encode(build_prompt(record.fields))
        response = tokenizer.encode(build_response(record.story))
        if len(prompt) + len(response) <= max_len + 1:  # inputs are all tokens but the last
            examples.append((prompt + response, len(prompt)))
    return examples


def make_batch(examples, device="cpu", multiple=1, max_width=None):
    """Pad examples to equal length. Targets are the inputs shifted by one, with the prompt and
    the padding masked out so only response tokens contribute to the loss.

    The width is rounded up to a multiple of `multiple` but not beyond `max_width` (it is never
    less than the longest example): fewer distinct tensor shapes keep the GPU's memory allocator
    from fragmenting.
    """
    longest = max(len(ids) for ids, _ in examples) - 1
    width = -(-longest // multiple) * multiple
    if max_width is not None:
        width = min(width, max_width)
    width = max(width, longest)
    x = torch.zeros(len(examples), width, dtype=torch.long)
    y = torch.full((len(examples), width), IGNORE, dtype=torch.long)
    for row, (ids, n_prompt) in enumerate(examples):
        n = len(ids) - 1
        x[row, :n] = torch.tensor(ids[:-1])
        y[row, n_prompt - 1 : n] = torch.tensor(ids[n_prompt:])  # predictions of response tokens
    return x.to(device), y.to(device)


@torch.no_grad()
def validation_loss(model, examples, batch_size, device, max_width):
    model.eval()
    losses = []
    for i in range(0, len(examples), batch_size):
        x, y = make_batch(examples[i : i + batch_size], device, PAD_MULTIPLE, max_width)
        losses.append(model(x, y)[1].item())
    model.train()
    return sum(losses) / len(losses)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", default="checkpoints/base.pt", help="pretrained checkpoint")
    p.add_argument("--out", default="checkpoints/sft.pt")
    p.add_argument("--data-dir", default="data/instruct")
    p.add_argument("--url", default=INSTRUCT_URL)
    p.add_argument("--max-mb", type=int, default=None, help="use only the start of a big file")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--micro-batch-size", type=int, default=8, help="examples per forward pass")
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--dropout", type=float, default=0.1, help="regularisation while fine-tuning")
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--eval-interval", type=int, default=250)
    p.add_argument("--eval-examples", type=int, default=200)
    p.add_argument("--eval-prompts", type=int, default=40, help="stories to score at the end")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = args.device or pick_device()
    model, tokenizer = load_model(args.base, device)
    model.cfg.dropout = args.dropout  # the base model may have been trained without any
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = args.dropout
    model.train()
    block_size = model.cfg.block_size

    raw = Path(args.data_dir) / "input.txt"
    raw.parent.mkdir(parents=True, exist_ok=True)
    if not raw.exists():
        print(f"downloading {args.url}")
        download(args.url, raw, args.max_mb * 1_000_000 if args.max_mb else None)
    records = parse_records(raw.read_text(encoding="utf-8", errors="ignore"))
    if args.max_mb:
        records = records[:-1]  # the byte range cut the last record off mid-story
    train_records, val_records = split_records(records)
    train = build_examples(train_records, tokenizer, block_size)
    val = build_examples(val_records, tokenizer, block_size)[: args.eval_examples]
    print(
        f"device {device} | {model.num_parameters():,} parameters | {len(train):,} train and "
        f"{len(val):,} validation examples ({len(records) - len(train) - len(val):,} skipped)"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    best_loss, best_state, start = float("inf"), None, time.time()
    running, counted = 0.0, 0
    before = validation_loss(model, val, 16, device, block_size)
    print(f"before fine-tuning: validation loss {before:.3f}")
    for step in range(args.steps + 1):
        if step % args.eval_interval == 0 or step == args.steps:
            loss = validation_loss(model, val, 16, device, block_size)
            if loss < best_loss:
                best_loss = loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            train_loss = running / max(1, counted)
            running, counted = 0.0, 0
            print(
                f"step {step:5d} | train loss {train_loss:.3f} | val loss {loss:.3f} "
                f"| {time.time() - start:6.1f}s"
            )
            if device == "mps":
                torch.mps.empty_cache()
        if step == args.steps:
            break
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, args.steps, args.lr, args.warmup)
        picks = torch.randint(len(train), (args.batch_size,)).tolist()
        optimizer.zero_grad(set_to_none=True)
        for i in range(0, args.batch_size, args.micro_batch_size):  # gradient accumulation
            chunk = [train[j] for j in picks[i : i + args.micro_batch_size]]
            x, y = make_batch(chunk, device, PAD_MULTIPLE, block_size)
            loss = model(x, y)[1]
            (loss * len(chunk) / args.batch_size).backward()
            running += loss.item() * len(chunk) / args.batch_size  # this step's mean loss, in parts
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        counted += 1

    model.load_state_dict(best_state)  # keep the checkpoint with the best validation loss
    model.eval()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": model.cfg.to_dict(),
            "model": model.state_dict(),
            "tokenizer_merges": [list(pair) for pair in tokenizer.merges],
            "sft": {"val_loss": best_loss, "steps": args.steps},
        },
        out,
    )
    print(f"saved {out} (best validation loss {best_loss:.3f})")

    scores = evaluate_following(model, tokenizer, val_records, args.eval_prompts)
    print("held-out requests:", format_scores(scores))
    fields = next(r.fields for r in val_records if "Words" in r.fields)
    story, _ = respond(model, tokenizer, fields)
    print(f"\n--- sample ---\n{build_prompt(fields)}{story}")


if __name__ == "__main__":
    main()
