# mylm

A small GPT-style language model written from scratch in PyTorch: byte-level BPE tokenizer,
decoder-only Transformer, training loop and text generation, with no dependencies beyond `torch`.

The default configuration (873k parameters) trains on Tiny Shakespeare in under two minutes on an
Apple Silicon GPU and produces text in the style of the plays.

## Quick start

Requires Python 3.10+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

mylm-prepare                              # download data, train tokenizer, write train/val splits
mylm-train                                # train; prints loss as it goes, saves checkpoints/model.pt
mylm-generate --prompt "ROMEO:"           # sample text from the trained model
```

`mylm-train` selects CUDA, Apple MPS or CPU automatically (override with `--device`). All options:
`mylm-train --help`.

### TinyStories

[TinyStories](https://arxiv.org/abs/2305.07759) is a corpus of simple, LLM-written children's
stories designed for small models. It gives much more coherent output than Shakespeare:

```bash
mylm-prepare --dataset tinystories        # ~19 MB, 2048-token vocabulary, writes data/tinystories/
mylm-train --data-dir data/tinystories --out checkpoints/tinystories.pt \
           --dim 256 --layers 6 --heads 8 --dropout 0.1
```

## Results

Default settings, 6000 steps, batch 32 x 128 tokens, Apple Silicon (MPS):

| Step | Train loss | Validation loss |
| ---: | ---------: | --------------: |
|    0 |      6.263 |           6.259 |
| 2000 |      2.819 |           3.149 |
| 4000 |      2.575 |           2.986 |
| 6000 |      2.507 |           2.904 |

The starting loss equals `ln(512)`, i.e. uniform guessing over the vocabulary. Sample output:

```
ROMEO:
Ay, help, away with her every gracious soul!

MERCUTIO:
Marry, sir, I say you, must stay, come.
```

The model learns the format of a play and plausible English phrasing. It does not produce
coherent sentences; that is expected for under 1M parameters and 0.5M training tokens.

### TinyStories

6000 steps, batch 32 x 128 tokens, 4.7M training tokens, vocabulary 2048:

| Model | Parameters | Training time | Validation loss | Loss per character |
| --- | ---: | ---: | ---: | ---: |
| Shakespeare, default (vocab 512) | 873k | ~2 min | 2.904 | 1.487 |
| TinyStories, `--dim 128 --layers 4` | 1.07M | ~2 min | 2.618 | 0.715 |
| TinyStories, `--dim 256 --layers 6` | 5.29M | ~6 min | **2.129** | 0.582 |
| same, `--norm rmsnorm` | 5.29M | ~6.7 min | 2.131 | 0.582 |
| same, `--pos rope` | 5.26M | ~6.5 min | **2.072** | 0.566 |

Loss is measured per token, and different tokenizers produce different numbers of characters per
token (1.95 for Shakespeare, 3.66 for TinyStories), so per-token losses are only comparable when
the dataset and tokenizer are the same. Dividing by characters per token gives the last column,
which is comparable across datasets. Sample from the largest model:

```
Anna and Ben are friends who are twins. They like to play with their toy cars and puzzle.
But today, they are bored. They ask their mom to help them pack up and have some fun.
```

The text is grammatical and follows a story arc but still contains invented words and drifts off
topic. Training loss (1.92) is below validation loss (2.13), so the model has begun to memorise.

## How it works

```
 data/input.txt ──► BPETokenizer ──► token IDs ──► get_batch ──► GPT ──► loss ──► AdamW
   (raw text)       tokenizer.py     train.pt      data.py     model.py          train.py
                                     val.pt
```

| Module | Responsibility |
| --- | --- |
| `mylm/tokenizer.py` | Byte-level BPE: `train`, `encode`, `decode`, `save`, `load` |
| `mylm/prepare.py` | Downloads the dataset, trains the tokenizer, writes tokenized train/validation splits |
| `mylm/data.py` | `get_batch`: random `(input, target)` windows where target is input shifted by one token |
| `mylm/model.py` | `GPT`: token and position embeddings, Transformer blocks, output head, `generate` |
| `mylm/train.py` | Training loop: AdamW, warmup and cosine learning rate, gradient clipping, evaluation |
| `mylm/generate.py` | Loads a checkpoint and samples text (temperature, top-k) |

See [docs/architecture.md](docs/architecture.md) for a walkthrough of each component.

## Tests

```bash
pytest
ruff check . && ruff format --check .
```

Tests cover tokenizer round trips (including non-ASCII text), batch construction, model shapes,
the causal mask (changing a token never affects earlier positions), initial loss, the ability to
overfit a single batch, and generation.

## Data and references

- Datasets: Tiny Shakespeare (about 1.1 MB), from Andrej Karpathy's
  [char-rnn](https://github.com/karpathy/char-rnn) repository; TinyStories (Eldan and Li, 2023),
  validation split, from [Hugging Face](https://huggingface.co/datasets/roneneldan/TinyStories).
- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762), 2017.
- Radford et al., *Language Models are Unsupervised Multitask Learners* (GPT-2), 2019.
- Sennrich et al., [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909), 2016 (BPE).

## License

MIT. See [LICENSE](LICENSE).
