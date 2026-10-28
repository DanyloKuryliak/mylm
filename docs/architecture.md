# Architecture

## 1. Tokenizer (`mylm/tokenizer.py`)

Neural networks operate on numbers, so text is converted to integer token IDs.

The tokenizer is byte-level BPE (byte-pair encoding):

1. Start with 256 tokens, one per possible byte, so any text can be represented.
2. Split the text into word-like chunks (a word with its leading space, a run of digits, a run of
   punctuation). Merges never cross chunk boundaries.
3. Repeatedly find the most frequent adjacent pair of tokens and replace it with a new token.
   Repeat until the vocabulary reaches the target size (512 by default).

Encoding applies the learned merges in the order they were learned. Decoding concatenates each
token's bytes. The tokenizer is trained only on the training split, so validation text stays unseen.

## 2. Data (`mylm/prepare.py`, `mylm/data.py`)

The token stream is split 90% / 10% into training and validation. Training examples come from
sliding windows over the stream: for a window `x` of `block_size` tokens, the target `y` is the same
window shifted one token to the right. At every position the model predicts the next token, so a
window yields `block_size` training signals and the text supplies its own labels.

### Choosing a dataset and vocabulary size

`mylm-prepare --dataset` selects the corpus; each has its own default vocabulary size.
Vocabulary size trades sequence length against embedding size: a larger vocabulary packs more
characters into each token (1.95 characters per token at 512, 3.66 at 2048 on TinyStories), so the
same context window covers more text, but the embedding and output matrices grow with it.

Loss is averaged per token, so it depends on the tokenizer. To compare runs with different
tokenizers or datasets, divide the loss by the average characters per token. Within one dataset and
tokenizer, per-token loss is directly comparable, which is how architecture changes are evaluated.

## 3. Model (`mylm/model.py`)

Input of shape `(batch, time)` of token IDs becomes logits of shape `(batch, time, vocab)`.

1. **Embeddings.** A token embedding maps each ID to a vector. By default a learned position
   embedding is added, because attention alone has no notion of order (see *Positions* below for
   the `--pos rope` alternative).
2. **Transformer blocks** (repeated `n_layers` times), each with two residual sub-layers:
   - **Causal multi-head self-attention.** Each token forms a query, key and value. Attention
     weights are `softmax(q·kᵀ / √d)` with a lower-triangular mask so a token only attends to
     itself and earlier tokens. Heads run in parallel and are mixed by an output projection.
   - **Feed-forward network.** Per token: expand 4x, GELU, project back.

   Both sub-layers use pre-LayerNorm and add their output to their input (`x = x + f(norm(x))`).
3. **Output head.** Final LayerNorm, then a linear layer to vocabulary logits. Its weights are
   shared with the token embedding (weight tying).

**Normalization.** `--norm layernorm` (default) subtracts the mean, divides by the standard
deviation and applies a learned scale and shift. `--norm rmsnorm` only divides by the
root-mean-square, `x / sqrt(mean(x²) + ε) · γ`, with no mean subtraction and no shift. On
TinyStories with the 5.3M-parameter model both reach the same validation loss (2.129 vs 2.131,
within seed noise), so the extra centering step is unnecessary. The RMSNorm here is written with
plain tensor operations and ran about 10% slower than PyTorch's fused `nn.LayerNorm`.

**Positions.** `--pos learned` (default) adds a trained vector per position, so the model must
learn that position 57 is next to position 56 and has no vector for positions past `block_size`.
`--pos rope` (rotary embeddings) instead rotates each query and key vector, viewed as
`head_dim / 2` two-dimensional points, by an angle `position * theta_i`, where `theta_i` shrinks
geometrically with the pair index. The dot product of a rotated query and key then depends only
on the distance between the two tokens, so relative position is built in and the position table
disappears. Values are not rotated: position only affects where attention looks, not what is
passed along. On TinyStories with the 5.3M-parameter model RoPE reaches validation loss 2.072
against 2.129, ahead at every evaluation, at about 6% more time per step.

**Grouped-query attention.** By default every query head has its own key and value head. With
`--kv-heads N` (N must divide `--heads`) there are still `--heads` query heads but only N key/value
heads, and each group of `heads / N` query heads shares one: query head `h` reads key/value head
`h // (heads / N)`. The KV cache stores only N heads, so it is `heads / N` times smaller (with 8
heads: 1.5 MiB, 0.375 MiB and 0.19 MiB for 8, 2 and 1 KV heads over a 128-token context), and the
key/value projections shrink too. On TinyStories with the 5.3M-parameter model, 2 KV heads reach
validation loss 2.145 and 1 KV head 2.145, against 2.129 with 8: a small cost, partly because the
model also has 11% fewer parameters. The benefit is for large models, where the cache dominates
memory; at this size the cache is already tiny.

**Feed-forward variants.** `--ff gelu` (default) is `down(gelu(up(x)))` with a 4x hidden width.
`--ff swiglu` is `down(silu(gate(x)) * up(x))`: two parallel projections multiplied element-wise,
so the gate path decides how much of each hidden feature passes. It has three matrices instead of
two, so the hidden width is about 8/3 of `dim` to keep the parameter count within 1%. At this
scale (5.3M parameters, 6000 steps) it did not help: validation loss 2.152 against 2.129. Reported
gains for gated MLPs come from much larger models, so it may not transfer down.

Dropout is applied to embeddings, attention weights and sub-layer outputs. Without it, the model
overfits the small dataset: validation loss starts rising while training loss keeps falling.

**How these options were compared.** Every variant was trained with the same data, tokenizer, step
count, learning-rate schedule and seed, changing one thing at a time. Repeating the baseline and
the RoPE run with a second seed moved the validation loss by about 0.01 (2.129 and 2.119; 2.072
and 2.061), so differences below roughly 0.01 are noise. Only RoPE's improvement (about 0.06) is
well outside that. The final base model therefore uses RoPE and leaves the other options at their
defaults.

## 4. Training (`mylm/train.py`)

- Loss: cross-entropy between predicted next-token distribution and the true next token.
- Optimizer: AdamW with weight decay 0.1.
- Learning rate: linear warmup for 100 steps, then cosine decay to 10% of the peak.
- Gradient clipping at norm 1.0.
- Every `--eval-interval` steps, loss is estimated on several random batches of both splits.
- The checkpoint stores the model config, weights and tokenizer merges, so it is self-contained.

## 5. Generation (`GPT.generate`)

Repeat: run the model, take the logits for the final position, divide by `temperature`, keep only
the `top_k` highest, sample one token from the softmax, append it.

**KV cache.** In a causal model the keys and values of a token depend only on the tokens before
it, so they never change as the text grows. Each attention layer therefore returns its keys and
values (`GPT.step` returns them for all layers), and the next step feeds only the newest token:
its query attends to the cached keys and values plus its own, with the causal mask offset by the
cache length. Positions follow the same offset: the learned embedding is looked up at
`offset .. offset + T`, and RoPE rotates by those angles. Work per step drops from the whole
context to one token.

When the text exceeds `block_size`, the model sees only the last `block_size` tokens, so every
token's position changes and the cache no longer matches. Those steps recompute the window, which
keeps the output identical to the uncached path (checked in the tests) but gives no speed-up.

`generate` also accepts a `stop` function that is called after every new token; chat uses it to end
a story as soon as the end-of-text marker appears.

## 6. Supervised fine-tuning (`mylm/instruct.py`, `mylm/sft.py`)

A pretrained model only continues text. Fine-tuning on request and answer pairs teaches it to
respond to a request instead.

**Data.** TinyStories-Instruct has one entry per story with some of these fields: `Words` (words
the story must use), `Features` (such as `Dialogue` or `BadEnding`), `Random sentence` (a sentence
to include), `Summary` and the `Story` itself. The order of the fields varies and the summary often
comes after the story, so `parse_records` splits on the field labels. Each entry becomes a request
(its fields except the story) and an answer (the story).

**Prompt format.** The fields are rendered in a fixed order:

```
### Instruction:
Write a short story.
Words: dog, ball
Features: Dialogue
Summary: A dog loses his ball and a girl helps him find it

### Response:
<story>
<|endoftext|>
```

The tokenizer has no special tokens, so the end marker is ordinary text; generation stops when it
appears in the output.

**Loss mask.** Targets are the inputs shifted by one token, as in pretraining, but the targets of
the prompt positions and of the padding are set to `-100`, which `cross_entropy` ignores. Only the
story and the end marker contribute to the loss, so the model learns to answer requests and not to
write them. Examples longer than the context window (about 6% of the entries) are dropped.

**Training details.**
- The last 500 entries are held out and never trained on; they provide the validation loss and the
  requests used by the evaluation.
- Each step accumulates gradients over micro-batches (`--micro-batch-size`), giving the same update
  as one batch of `--batch-size` with a fraction of the activation memory.
- Batch widths are rounded up to a multiple of 64, so the GPU sees only a few distinct tensor
  shapes. A first run with full batches of arbitrary widths ran out of GPU memory on a machine
  shared with other programs; micro-batches and bucketed widths together fixed it (the two
  changes were not tested separately).
- Dropout is switched back on (`--dropout`, default 0.1) because the base model is pretrained
  without it and a few passes over the fine-tuning data would otherwise overfit.
- The checkpoint with the best validation loss is the one that is saved.

## 7. Evaluation (`mylm/evaluate.py`)

`mylm-eval` generates a story for held-out requests that contain a `Words` field and reports:

- **word coverage**: the average fraction of the requested words that appear in the story as whole
  words (case-insensitive);
- **all words used**: the share of stories that use every requested word;
- **ended on its own**: the share of stories that wrote the end marker within the token limit;
- **human story**: the word coverage of the dataset's own story, the best an answer can reasonably
  score. It is about 0.9 and not 1.0 because authors often inflect a word ("blink" becomes
  "blinked") and the check requires the exact word.

The metric is cheap and checkable by code, but it measures one thing. A story can use every word and
still be incoherent, and the share that ended on its own is not informative on its own: the
pretrained model also ends its stories without having seen any request.

## 8. Chat (`mylm/chat.py`)

`mylm-chat` turns what is typed into a request. Plain text becomes the `Summary` field. Labelled
fields can be typed on one line and separated with `|`, for example
`Words: dog, ball | Features: Dialogue`. The request is rendered with the prompt format above, the
model generates until it writes the end marker or reaches `--max-new-tokens`, and the story is
printed. Each request is independent: the model keeps no conversation history.

## Configurations

| Setting | Defaults (`mylm-train`) | Base model used for fine-tuning |
| --- | --- | --- |
| Dataset | Tiny Shakespeare | `tinystories-large` (73M training tokens) |
| Vocabulary | 512 | 4096 |
| Context length | 128 | 512 |
| Width / layers / heads | 128 / 4 / 4 | 384 / 8 / 6 |
| Positions | learned | RoPE |
| Parameters | 873,216 | 15,757,056 |
| Dropout | 0.2 | 0.0 |
| Batch x steps | 32 x 128 tokens, 6000 | 16 x 512 tokens, 10,000 |
| Peak learning rate | 3e-3 | 1e-3 |
