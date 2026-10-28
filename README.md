# mylm

A small GPT-style language model written from scratch in PyTorch, from the tokenizer to a chat
command: byte-level BPE tokenizer, decoder-only Transformer with selectable modern components
(RoPE, RMSNorm, SwiGLU, grouped-query attention), a KV cache, pretraining, supervised
fine-tuning on instructions, an evaluation and an interactive chat. The only dependency is `torch`.

Pretrained on 73M tokens of children's stories (a 15.8M-parameter model), then fine-tuned to write
a story from a request such as "use the words *dog* and *ball*, with dialogue":

```
you> Words: dog, ball | Features: Dialogue | Summary: A dog loses his ball and a girl helps him find it
Once upon a time, there was a big, red ball. The ball was in a park and the flowers had many flowers.
One day, a little dog saw the ball and ran after it. [...]
The girl said, "Don't worry, I'll help you find it." She picked up the ball in her hands [...]
```

The model is small, so its stories are simple and its plots wander; see [Limitations](#limitations).

## Quick start

Requires Python 3.10+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

### 1. A first model on Shakespeare (about 2 minutes)

```bash
mylm-prepare                              # download data, train tokenizer, write train/val splits
mylm-train                                # prints loss as it goes, saves checkpoints/model.pt
mylm-generate --prompt "ROMEO:"           # sample text from the trained model
```

`mylm-train` selects CUDA, Apple MPS or CPU automatically (override with `--device`). All options:
`mylm-train --help`.

### 2. The full pipeline: pretrain, fine-tune, chat

```bash
# Pretrain a 15.8M-parameter base model on 73M tokens of TinyStories (about 50 min on Apple Silicon)
mylm-prepare --dataset tinystories-large
mylm-train --data-dir data/tinystories-large --out checkpoints/base.pt \
           --dim 384 --layers 8 --heads 6 --block-size 512 --batch-size 16 \
           --dropout 0.0 --pos rope --lr 1e-3 --steps 10000 --eval-interval 500

# Fine-tune on about 70k story requests from TinyStories-Instruct (about 30 min)
mylm-sft --base checkpoints/base.pt --out checkpoints/sft.pt --data-dir data/instruct-large \
         --url https://huggingface.co/datasets/roneneldan/TinyStoriesInstruct/resolve/main/TinyStories-Instruct-train.txt \
         --max-mb 80 --steps 6000

# Talk to it
mylm-chat --checkpoint checkpoints/sft.pt
mylm-chat --prompt "a little dragon who is afraid of fire"     # one request, then exit

# Score it on held-out requests (and compare with the base model)
mylm-eval --checkpoint checkpoints/sft.pt --data-dir data/instruct-large -n 40
mylm-eval --checkpoint checkpoints/base.pt --data-dir data/instruct-large -n 40
```

`mylm-sft` without `--url` uses the small validation file of TinyStories-Instruct (25k requests,
27 MB), which is enough to try the pipeline quickly. If GPU memory is short, lower
`--micro-batch-size`.

In `mylm-chat`, plain text is treated as a summary of the wanted story. Labelled fields
(`Words`, `Features`, `Random sentence`, `Summary`) can be given on one line separated by `|`.
Each request is independent; the model keeps no conversation history.

## Results

### Pretraining and fine-tuning

| Model | Parameters | Validation loss |
| --- | ---: | ---: |
| Base, pretrained on `tinystories-large` (vocabulary 4096) | 15.76M | 1.619 |
| After fine-tuning (loss on the story tokens of held-out requests) | 15.76M | 1.494 before, **1.150** after |

Whether the story uses the requested words, on 40 held-out requests (`mylm-eval`, temperature 0.8):

| | Word coverage | Stories using all words |
| --- | ---: | ---: |
| Base model, same prompt | 0.09 | 0% |
| **After fine-tuning** | **0.66** | **35%** |
| Human-written story (reference) | 0.92 | |

The base model ignores the request and writes an unrelated story. After fine-tuning it follows the
summary, includes the requested sentence and uses most of the requested words. A sample for the
request `Words: own, paper, dangerous / Features: Dialogue, Twist / Random sentence: He loved to draw
on paper with his crayons.`:

```
One day, a boy named Tim found a paper. The paper was very big and red. Tim loved to draw on paper
with his crayons. He would draw a big, red house and a huge dragon on the paper. [...]
His mom said, "Yes, but be careful." Tim drew a big, red dragon inside the paper. The dragon looked
at the paper and said, "Hi, I am a dragon! I want to be your friend." [...]
```

It used "paper" and the required sentence but not "own" or "dangerous".

### Architecture options

Each option was compared with the baseline on the 19 MB `tinystories` dataset (4.7M training
tokens, vocabulary 2048), 6000 steps, batch 32 x 128 tokens, model of width 256 with 6 layers and
8 heads. Only the stated option differs. A second seed moved the baseline by 0.010 and RoPE by
0.011, so differences below about 0.01 are noise.

| Setting | Parameters | Validation loss, seed 1337 | seed 2 |
| --- | ---: | ---: | ---: |
| Baseline (LayerNorm, learned positions, GELU, 8 KV heads) | 5.29M | 2.129 | 2.119 |
| `--norm rmsnorm` | 5.29M | 2.131 | |
| `--pos rope` | 5.26M | **2.072** | **2.061** |
| `--ff swiglu` | 5.27M | 2.152 | |
| `--kv-heads 2` | 4.70M | 2.145 | |
| `--kv-heads 1` | 4.60M | 2.145 | |
| `--pos rope --norm rmsnorm` | 5.25M | 2.077 | |
| `--pos rope --norm rmsnorm --ff swiglu --kv-heads 2` | 4.64M | 2.090 | |

- **RoPE** is the only clear improvement, so the base model uses it.
- **RMSNorm** makes no measurable difference here, and the version in this repository is built from
  plain tensor operations and ran about 10% slower than PyTorch's fused `LayerNorm`.
- **SwiGLU** and **grouped-query attention** cost a little at this size. They are designed for
  large models: GQA shrinks the KV cache by 4x with 2 KV heads and 8x with 1, which matters when the
  cache dominates memory, and not for a 5M-parameter model.

For comparison across datasets, loss per token must be divided by the characters per token, which
depends on the tokenizer (1.95 for Shakespeare at vocabulary 512, 3.66 for TinyStories at 2048,
4.01 for `tinystories-large` at 4096):

| Model | Validation loss | Loss per character |
| --- | ---: | ---: |
| Shakespeare, default (873k parameters) | 2.904 | 1.487 |
| TinyStories, width 128 with 4 layers (1.07M) | 2.618 | 0.715 |
| TinyStories, width 256 with 6 layers (5.29M) | 2.129 | 0.582 |

### Generation speed (KV cache)

`mylm-generate` reuses each layer's keys and values from earlier steps instead of recomputing the
whole context for every new token; `--no-cache` turns this off. The text is identical either way
(tested). Measured on the 5.3M-parameter RoPE model:

| Device | New tokens | No cache | Cache | Speed-up |
| --- | ---: | ---: | ---: | ---: |
| CPU | 100 | 0.22 s | 0.07 s | 3.0x |
| CPU | 300 | 0.84 s | 0.64 s | 1.3x |
| Apple MPS | 100 | 0.48 s | 0.79 s | 0.6x |
| Apple MPS | 300 | 0.86 s | 0.90 s | 1.0x |

The cache only helps while the text fits in the context window; after that the window slides,
positions shift and the cache is rebuilt at every step. On the GPU this model is too small for the
saved arithmetic to outweigh the per-step overhead of extending the cache.

## How it works

```
text ──► BPETokenizer ──► token IDs ──► get_batch ──► GPT ──► next-token loss ──► AdamW   (pretraining)
          tokenizer.py     prepare.py     data.py    model.py       train.py
                                                        │
                       TinyStories-Instruct ──► request + story pairs ──► loss on story only   (sft.py)
                                                        │
                                  mylm-chat / mylm-eval: prompt format, generation, scoring
```

| Module | Responsibility |
| --- | --- |
| `mylm/tokenizer.py` | Byte-level BPE: `train`, `encode`, `decode`, `save`, `load` |
| `mylm/prepare.py` | Downloads a dataset, trains the tokenizer, writes tokenized train/validation splits |
| `mylm/data.py` | `get_batch`: random `(input, target)` windows where target is input shifted by one token |
| `mylm/model.py` | `GPT` and its parts: attention (RoPE, grouped-query), feed-forward variants, norms, KV cache, `generate` |
| `mylm/train.py` | Pretraining loop: AdamW, warmup and cosine learning rate, gradient clipping, evaluation |
| `mylm/generate.py` | Samples text from a checkpoint (temperature, top-k) |
| `mylm/instruct.py` | Request format: parsing TinyStories-Instruct, prompt template, word coverage, `respond` |
| `mylm/sft.py` | Supervised fine-tuning with the loss restricted to the answer |
| `mylm/evaluate.py` | Scores a checkpoint on held-out requests |
| `mylm/chat.py` | Interactive chat and one-shot requests |
| `mylm/checkpoint.py` | Loads a model and its tokenizer from a checkpoint |

See [docs/architecture.md](docs/architecture.md) for a walkthrough of each component and the
reasoning behind the options.

## Limitations

- The model has 15.8M parameters and a 512-token context. It writes simple children's stories in
  the style of its training data; it does not know facts and cannot answer general questions.
- It often misses one of the requested words (35% of stories use all of them), and plots drift,
  repeat or contradict themselves.
- The evaluation checks only whether requested words appear. It says nothing about whether a story
  is good.
- Architecture comparisons use one or two seeds on a 5M-parameter model, so only differences larger
  than about 0.01 validation loss mean anything, and conclusions may not transfer to larger models.

## Tests

```bash
pytest
ruff check . && ruff format --check .
```

Tests cover tokenizer round trips (including non-ASCII text), batch construction, dataset
preparation, model shapes, the causal mask, each architecture option (norm, RoPE, SwiGLU,
grouped-query attention), equality of cached and uncached generation, parsing of the instruction
data, the loss mask used in fine-tuning, and word coverage.

## Data and references

- Datasets: Tiny Shakespeare (about 1.1 MB), from Andrej Karpathy's
  [char-rnn](https://github.com/karpathy/char-rnn) repository;
  [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) and
  [TinyStories-Instruct](https://huggingface.co/datasets/roneneldan/TinyStoriesInstruct)
  (Eldan and Li, 2023) from Hugging Face.
- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762), 2017.
- Radford et al., *Language Models are Unsupervised Multitask Learners* (GPT-2), 2019.
- Sennrich et al., [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909), 2016 (BPE).
- Eldan and Li, [TinyStories](https://arxiv.org/abs/2305.07759), 2023.
- Su et al., [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864), 2021.
- Zhang and Sennrich, [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467), 2019.
- Shazeer, [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202), 2020.
- Ainslie et al., [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245), 2023.

## License

MIT. See [LICENSE](LICENSE).
