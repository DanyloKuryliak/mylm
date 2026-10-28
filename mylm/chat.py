"""Chat with a fine-tuned model: describe the story you want and it writes one."""

import argparse

import torch

from mylm.checkpoint import load_model
from mylm.instruct import parse_request, respond
from mylm.train import pick_device

BANNER = """Describe a story and the model will write it. You can also give labelled requests,
separated by "|", for example:  Words: dog, ball | Features: Dialogue | Summary: A dog loses a ball
Type /quit to leave."""


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="checkpoints/sft.pt")
    p.add_argument("--prompt", help="answer this one request and exit instead of chatting")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--max-new-tokens", type=int, default=450)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
    device = args.device or pick_device()
    model, tokenizer = load_model(args.checkpoint, device)

    def answer(text):
        fields = parse_request(text.replace("|", "\n"))
        story, finished = respond(
            model, tokenizer, fields, args.max_new_tokens, args.temperature, args.top_k, device
        )
        print(story + ("" if finished else "\n[cut off: raise --max-new-tokens]"))

    if args.prompt:
        answer(args.prompt)
        return
    print(BANNER)
    while True:
        try:
            text = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if text in ("/quit", "/exit"):
            break
        if text:
            answer(text)


if __name__ == "__main__":
    main()
