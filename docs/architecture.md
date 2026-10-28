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

Dropout is applied to embeddings, attention weights and sub-layer outputs. Without it, the model
overfits the small dataset: validation loss starts rising while training loss keeps falling.

## 4. Training (`mylm/train.py`)

- Loss: cross-entropy between predicted next-token distribution and the true next token.
- Optimizer: AdamW with weight decay 0.1.
- Learning rate: linear warmup for 100 steps, then cosine decay to 10% of the peak.
- Gradient clipping at norm 1.0.
- Every `--eval-interval` steps, loss is estimated on several random batches of both splits.
- The checkpoint stores the model config, weights and tokenizer merges, so it is self-contained.

## 5. Generation (`GPT.generate`)

Repeat: run the model on the last `block_size` tokens, take the logits for the final position,
divide by `temperature`, keep only the `top_k` highest, sample one token from the softmax, append it.

## Default configuration

| Setting | Value |
| --- | --- |
| Vocabulary | 512 |
| Context length | 128 |
| Width / layers / heads | 128 / 4 / 4 |
| Parameters | 873,216 |
| Dropout | 0.2 |
