# Qwen3 Inference from Scratch

A from-scratch PyTorch implementation of Qwen3 dense model inference. Supports all Qwen3 dense models from 0.6B to 32B

Target model: [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)

No `transformers` library — just raw tensor operations. Read [blog post](https://blog.yellowday.day/posts/qwen3_from_scratch/) for details behind.

## Quick Start

All commands run from the **project root** (`llm-from-scratch/`):

```bash
# 1. Download model checkpoint (default: Qwen3-0.6B)
bash scripts/download.sh Qwen/Qwen3-0.6B

# 2. Run inference
uv run python -m qwen3.main

# Pick a different model
uv run python -m qwen3.main -m Qwen3-4B -p "Explain quantum computing"

# Tune generation parameters
uv run python -m qwen3.main -m Qwen3-1.7B -t 0.7 -k 50 -n 512

# Enable thinking mode
uv run python -m qwen3.main --thinking -p "Which one is bigger? 9.9 or 9.11"
```

### CLI Options

| Flag | Description | Default |
|---|---|---|
| `-m`, `--model` | `Qwen3-0.6B`, `Qwen3-1.7B`, `Qwen3-4B`, `Qwen3-8B`, `Qwen3-14B`, `Qwen3-32B` | `Qwen3-0.6B` |
| `-p`, `--prompt` | User prompt text | `Which is bigger, 9.9 or 9.11?` |
| `-t`, `--temperature` | Sampling temperature (0 = greedy) | `1.0` |
| `-k`, `--top-k` | Top-k filtering (-1 = disabled) | `-1` |
| `-n`, `--max-tokens` | Max new tokens to generate | `1024` |
| `--thinking` | Enable thinking mode | off |
| `-d`, `--device` | `cuda`, `cpu`, or `auto` | `auto` |

If a checkpoint is missing, you'll be prompted to download it automatically (requires `huggingface_hub`).

## Architecture Overview

```
Token IDs → Embedding → [28 × Transformer Block] → RMSNorm → LM Head → Logits → Sample → Token
                              │
                              ├── RMSNorm
                              ├── GQA (Grouped Query Attention) + Residual
                              ├── RMSNorm
                              └── SwiGLU FFN + Residual
```

## Model Config (Qwen3-0.6B)

| Param | Value | Meaning |
|---|---|---|
| `vocab_size` | 151936 | tokenizer vocabulary size |
| `emb_dim` | 1024 | hidden dimension (d_model) |
| `n_heads` | 16 | query attention heads |
| `n_kv_groups` | 8 | key/value heads (GQA) |
| `head_dim` | 128 | per-head dimension |
| `n_layers` | 28 | transformer blocks |
| `hidden_dim` | 3072 | FFN intermediate size |
| `rope_base` | 1,000,000 | RoPE frequency base |
| `context_length` | 40960 | max sequence length |

Note: `n_heads * head_dim = 2048`, not `emb_dim (1024)`. The Q/K/V projections change dimensionality.

## File Structure

```
qwen3/
├── config.py       # Step 1: model config dataclass
├── tokenizer.py    # Step 2: tokenizer + chat template
├── layers.py       # Steps 3-7: RMSNorm, RoPE, GQA, SwiGLU FFN, Transformer Block
├── model.py        # Step 8: full model (stack blocks + LM head)
├── weights.py      # Step 9: load safetensors weights
├── generate.py     # Step 10: generation loop with KV cache
└── main.py         # tie it all together

tests/qwen3/
├── test_knowledge.py # integration tests (single + batch)
└── test_accuracy.py  # token-level accuracy vs HuggingFace transformers
```

## Learning Guide

Follow these steps in order. Each step is independently testable.

### Step 1: Model Config (`config.py`)

Create a dataclass that loads hyperparameters from the HuggingFace `config.json`.

### Step 2: Tokenizer (`tokenizer.py`)

Qwen3 uses HuggingFace `tokenizers` library (not `transformers`). Load `tokenizer.json` from the checkpoint.

Chat template format:
```
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
<|think>

<|/think>

```

The `<|think>` / `<|/think>` tags are Qwen3's thinking mode. Even in non-thinking mode, include empty think tags.

### Step 3: RMSNorm (`layers.py`)

```
RMSNorm(x) = x / sqrt(mean(x²) + eps) * gamma
```

- No mean subtraction, no bias (unlike LayerNorm)
- Cast to float32 for norm computation to avoid bfloat16 numerical issues

### Step 4: Rotary Position Embeddings (`layers.py`)

Precompute cos/sin tables:
```
freqs[i] = 1.0 / (rope_base ^ (2i / head_dim))
angles[pos, i] = pos * freqs[i]
```

Apply rotation:
```
x1, x2 = split(x)
rotated = concat(-x2, x1)
output = x * cos + rotated * sin
```

Key: RoPE accepts a `position_ids` tensor of shape `(batch, seq_len)` — each sequence in a batch can have its own position mapping. This is essential for left-padded batches where padding positions don't correspond to real token positions.

### Step 5: Grouped Query Attention (`layers.py`)

```
x → Q, K, V projections
  → QK-Norm (RMSNorm per head)
  → RoPE on Q, K
  → KV cache concat
  → Repeat KV heads (8 → 16)
  → Scaled dot-product attention + causal mask
  → Output projection
```

Weight shapes:
- `W_q: [1024, 2048]`, `W_k: [1024, 1024]`, `W_v: [1024, 1024]`, `W_o: [2048, 1024]`

Supports an optional `attn_mask` (additive, 0 = attend, -inf = block) for combined causal + padding masking in batch mode. Falls back to auto causal mask for single-sequence inference.

### Step 6: SwiGLU Feed-Forward (`layers.py`)

```
FFN(x) = SiLU(x @ W_gate) * (x @ W_up) @ W_down
```

Three matrices at 3x expansion instead of two at 4x — similar total params.

### Step 7: Transformer Block (`layers.py`)

Each block: pre-norm → attention + residual → pre-norm → FFN + residual.

### Step 8: Assemble Model (`model.py`)

Stack N transformer blocks, add final RMSNorm and LM head.

### Step 9: Weight Loading (`weights.py`)

Map HuggingFace weight names to your model's state dict keys. HuggingFace stores linear weights as `[out, in]` — your `nn.Linear` handles this automatically.

### Step 10: Generation Loop (`generate.py`)

**Single sequence** (`generate()`):
1. **Prefill**: forward pass on full prompt, get KV cache
2. **Decode loop**: forward only the new token, reuse KV cache (grows by 1 each step)
3. **Sampling**: top-k filtering → temperature scaling → multinomial sampling

**Batch generation** (`generate_batch()`):
1. **Left-pad** variable-length prompts to the same length using `tokenizer.pad_token_id`
2. **Build `position_ids`**: padding positions get 0, real tokens get 0, 1, 2, ...
3. **Build attention mask**: combined causal + padding mask. Padding positions self-attend to avoid NaN from `softmax(all -inf)`
4. **Prefill + decode** with per-sequence EOS tracking
5. Batch output is semantically correct (bf16 rounding may cause minor token-level divergence from single-sequence)

## Tests

Run from the project root. Tests require downloaded checkpoints and `--model` flag.

```bash
# Knowledge tests: math, facts, thinking mode, batch correctness
uv run python -m pytest tests/qwen3/test_knowledge.py -m slow -v --model Qwen3-0.6B -s

# Accuracy tests: token-level match vs HuggingFace transformers
uv run python -m pytest tests/qwen3/test_accuracy.py -m slow -v --model Qwen3-0.6B -s
```

## Notes

**Accuracy comparison fixes (bf16 precision)**: When running in bfloat16, tiny rounding differences compound across 28+ transformer layers, causing token mismatches against HuggingFace transformers. Three fixes were needed to achieve exact token-level match:

1. **RMSNorm casting order** — Our original code multiplied the weight while still in float32 (`x / rms * weight` then `.to(dtype)`). HF casts the normalized value to bf16 *before* the weight multiply: `weight * x.to(dtype)`. The different rounding order accumulates across layers.

2. **RoPE buffer dtype** — Our cos/sin buffers were stored in float32 but the Q/K tensors are bf16. The mixed-precision multiply `bf16 * f32` produces different results from `bf16 * bf16`. Fixed by casting buffers to model dtype when indexing: `self.cos[position_ids].to(x.dtype)`.

3. **Softmax precision** — Our attention softmax operated in bf16 (`F.softmax(..., dtype=x.dtype)`). HF computes softmax in float32 for numerical stability: `F.softmax(..., dtype=torch.float32).to(x.dtype)`.

The test uses `attn_implementation="eager"` when loading HF models because SDPA uses fused GPU kernels with different floating-point accumulation paths. Eager mode uses the same manual attention math as our implementation, making the comparison meaningful. Use `scripts/compare_hf.py` for layer-level diagnostics.

**Weight tying vs `load_state_dict(assign=True)`**: When `tie_word_embeddings=True`, the model ties `lm_head.weight` to `tok_emb.weight` during init. However, `assign=True` replaces Parameter objects, breaking this tie. Some HuggingFace checkpoints (e.g. Qwen3-0.6B) redundantly include `lm_head.weight` so it gets loaded anyway, while others (e.g. Qwen3-4B) omit it — leaving `lm_head.weight` as an uninitialized meta tensor. Our `load_weights` detects this via `is_meta` and re-ties automatically.

## Dependencies

Managed by [uv](https://docs.astral.sh/uv/). Dependencies are declared in the root `pyproject.toml` and installed automatically on `uv run`.
