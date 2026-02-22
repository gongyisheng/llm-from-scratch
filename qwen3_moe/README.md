# Qwen3-MoE Inference from Scratch

A from-scratch PyTorch implementation of Qwen3-MoE (Mixture of Experts) inference. Builds on the [Qwen3 dense implementation](../qwen3/README.md) — same attention, tokenizer, and generation loop, but replaces the dense FFN with sparse expert routing.

Target model: [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) — 30B total params, ~3B active per token.

## Quick Start

All commands run from the **project root** (`llm-from-scratch/`):

```bash
# 1. Download model checkpoint (~61GB in bf16)
bash scripts/download.sh Qwen/Qwen3-30B-A3B

# 2. Run inference
uv run python -m qwen3_moe.main

# Custom prompt
uv run python -m qwen3_moe.main -p "Explain mixture of experts"

# Tune generation parameters
uv run python -m qwen3_moe.main -t 0.7 -k 50 -n 512

# Enable thinking mode
uv run python -m qwen3_moe.main --thinking -p "Which one is bigger? 9.9 or 9.11"
```

### CLI Options

| Flag | Description | Default |
|---|---|---|
| `-m`, `--model` | `Qwen3-30B-A3B` | `Qwen3-30B-A3B` |
| `-p`, `--prompt` | User prompt text | `Which is bigger, 9.9 or 9.11?` |
| `-t`, `--temperature` | Sampling temperature (0 = greedy) | `1.0` |
| `-k`, `--top-k` | Top-k filtering (-1 = disabled) | `-1` |
| `-n`, `--max-tokens` | Max new tokens to generate | `4096` |
| `--thinking` | Enable thinking mode | off |
| `-d`, `--device` | `cuda`, `cpu`, or `auto` | `auto` |

If a checkpoint is missing, you'll be prompted to download it automatically (requires `huggingface_hub`).

## What Changes from Qwen3 Dense to MoE

```
Qwen3 Dense                          Qwen3 MoE
─────────────                        ─────────────
TransformerBlock:                    MoETransformersBlock:
  RMSNorm → GQA → residual            RMSNorm → GQA → residual           (same)
  RMSNorm → SwiGLU FFN → residual     RMSNorm → SparseMoEBlock → residual  (CHANGED)
                                              │
                                              ├── MoEGate: picks 8 of 128 experts
                                              └── 128 × small SwiGLU experts
```

Only 3 things change:
1. **Config** — 4 new MoE fields (`n_experts`, `n_experts_per_token`, `moe_hidden_dim`, `moe_norm_topk_prob`)
2. **FFN replacement** — `SwiGLUFFN` → `SparseMoEBlock` (gate + 128 experts)
3. **Weight mapping** — new keys for gate and expert weights

Everything else (RMSNorm, RoPE, GQA, tokenizer, generation loop) is identical to Qwen3.

## Architecture Overview

```
Token IDs → Embedding → [48 × MoE Transformer Block] → RMSNorm → LM Head → Logits → Sample → Token
                              │
                              ├── RMSNorm
                              ├── GQA (Grouped Query Attention) + Residual
                              ├── RMSNorm
                              └── SparseMoEBlock + Residual
                                    │
                                    ├── MoEGate → top-8 of 128 experts
                                    └── 128 × SwiGLUFFN (small)
```

## Qwen3-30B-A3B Config

| Param | Value | vs Qwen3-0.6B |
|---|---|---|
| `emb_dim` (hidden_size) | 2048 | 1024 |
| `n_layers` | 48 | 28 |
| `n_heads` | 32 | 16 |
| `n_kv_groups` | 4 | 8 |
| `head_dim` | 128 | 128 |
| `n_experts` | 128 | — |
| `n_experts_per_token` | 8 | — |
| `moe_hidden_dim` | 768 | — |
| `moe_norm_topk_prob` | true | — |
| `tie_word_embeddings` | false | true |
| `vocab_size` | 151936 | 151936 |
| `context_length` | 40960 | 40960 |

Note: `tie_word_embeddings=false` means `lm_head` has its own weight matrix separate from `tok_emb` (Qwen3-0.6B ties them).

## How MoE Routing Works

The core idea: instead of one large FFN processing every token, route each token to a subset of smaller expert FFNs.

```
Token hidden state (2048-dim)
        │
        ▼
┌─── MoEGate ────┐
│ Linear(2048, 128) ← produces 128 scores, one per expert
│ Top-8 selection   ← pick 8 highest-scoring experts
│ Softmax over 8    ← normalize weights to sum to 1
└────────┬─────────┘
         │  routing_weights: (num_tokens, 8)
         │  selected_experts: (num_tokens, 8)
         ▼
┌─ For each of 128 experts ─┐
│  Find tokens routed here   │
│  Expert_i(tokens) × weight │  ← run through expert, scale by routing weight
└─────────┬──────────────────┘
          │
          ▼  Sum all expert outputs per token
     Output (2048-dim)
```

Each expert is a small SwiGLU FFN:
```
Expert_i(x) = SiLU(x @ gate_proj) * (x @ up_proj) @ down_proj

gate_proj: (2048, 768)   ← much smaller than dense FFN's (2048, 3072)
up_proj:   (2048, 768)
down_proj: (768, 2048)
```

Why it works: 128 experts × 768 intermediate = 98,304 total capacity, but only 8 × 768 = 6,144 active per token. The model learns to specialize experts for different token types.

## File Structure

```
qwen3_moe/
├── config.py       # Step 1: config dataclass with MoE fields
├── tokenizer.py    # (reused from qwen3, identical)
├── layers.py       # Steps 2-4: all layers (RMSNorm, RoPE, GQA, SwiGLUFFN, MoEGate, SparseMoEBlock, MoETransformersBlock)
├── model.py        # Step 5: full model assembly
├── weights.py      # Step 6: weight mapping for gate + experts
├── generate.py     # Step 7: generation loop (same logic as qwen3)
└── main.py         # CLI entry point
```

## Implementation Guide

Assumes you've already built the [Qwen3 dense model](../qwen3/README.md). Each step below focuses only on what's new for MoE.

### Step 1: Config (`config.py`)

Add 4 MoE-specific fields to your config dataclass, parsed from `config.json`:

```python
n_experts: int             # 128 — total expert count
n_experts_per_token: int   # 8 — experts activated per token
moe_hidden_dim: int        # 768 — each expert's FFN hidden dim
moe_norm_topk_prob: bool   # True — normalize routing weights after top-k
```

Also note `tie_word_embeddings=false` — unlike Qwen3-0.6B, `lm_head` has its own weights.

You can subclass `Qwen3Config` or create a standalone `Qwen3MoEConfig`. This implementation uses a standalone dataclass.

### Step 2: Reuse Dense Layers (`layers.py`)

Copy or import these unchanged from Qwen3: `RMSNorm`, `RoPE`, `GroupQueryAttention`, `SwiGLUFFN`.

The `SwiGLUFFN` class is reused as-is for each expert — the only difference is the intermediate dimension (`768` instead of `3072`).

### Step 3: MoE Gate (`layers.py`)

The gate decides which experts process each token. This is the key new component.

```python
class MoEGate(nn.Module):
    # proj: Linear(emb_dim, n_experts, bias=False)
    # i.e., Linear(2048, 128)
```

Forward logic:
1. **Score all experts**: `scores = proj(hidden_states)` → shape `(num_tokens, 128)`
2. **Top-k selection**: `torch.topk(scores, n_experts_per_token)` → top-8 values and indices
3. **Normalize**: `softmax(top_k_values)` → makes the 8 weights sum to 1.0

Key design choice: softmax is applied **after** top-k selection (controlled by `moe_norm_topk_prob=True`). This means only the 8 selected scores compete, not all 128.

Returns: `routing_weights (num_tokens, 8)` and `selected_experts (num_tokens, 8)`

### Step 4: SparseMoEBlock (`layers.py`)

Combines the gate with 128 expert FFNs. This replaces the single dense `SwiGLUFFN`.

```python
class SparseMoEBlock(nn.Module):
    # gate: MoEGate
    # experts: ModuleList of 128 SwiGLUFFN(emb_dim=2048, moe_hidden_dim=768)
```

Forward logic — the **loop-over-experts approach**:
```
1. Flatten input: (batch, seq, hidden) → (num_tokens, hidden)
2. Get routing: weights, expert_indices = gate(hidden_states)
3. Allocate output: zeros_like(hidden_states)
4. For each expert_idx in 0..127:
     - torch.where(selected_experts == expert_idx) → find routed tokens
     - If any: run those tokens through expert, multiply by routing weight
     - Accumulate into output via output[token_idx] += expert_output * weights
5. Reshape output back to (batch, seq, hidden)
```

Why loop over experts (not tokens)? Each expert processes a variable-sized batch of tokens. Looping over experts gives larger matrix multiplies than looping over tokens (where each runs 8 tiny matmuls).

### Step 5: Assemble Model (`model.py`)

Minimal change from Qwen3:
- Use `MoETransformersBlock` (which uses `SparseMoEBlock`) instead of the dense `TransformerBlock`
- 48 layers instead of 28
- `tie_word_embeddings=false` — `lm_head` has its own weight, no sharing with `tok_emb`

The `MoETransformersBlock` is identical to Qwen3's transformer block except the FFN:
```
norm1 → GQA → residual             (identical to Qwen3)
norm2 → SparseMoEBlock → residual  (FFN replaced with MoE)
```

### Step 6: Weight Mapping (`weights.py`)

HuggingFace weight names → your model's state dict. Same attention/norm mappings as Qwen3, plus:

```
# MoE gate (one per layer):
model.layers.{i}.mlp.gate.weight → layers.{i}.moe_ffn.gate.proj.weight

# MoE experts (128 per layer, 3 matrices each):
model.layers.{i}.mlp.experts.{j}.gate_proj.weight → layers.{i}.moe_ffn.experts.{j}.gate_proj.weight
model.layers.{i}.mlp.experts.{j}.up_proj.weight   → layers.{i}.moe_ffn.experts.{j}.up_proj.weight
model.layers.{i}.mlp.experts.{j}.down_proj.weight  → layers.{i}.moe_ffn.experts.{j}.down_proj.weight
```

The expert weights just need `mlp.experts.` → `moe_ffn.experts.` prefix replacement. The gate requires mapping `mlp.gate.weight` → `moe_ffn.gate.proj.weight`.

### Step 7: Generation Loop (`generate.py`)

Reuse from Qwen3 as-is. The generation loop doesn't care whether the model uses dense FFN or MoE — it just calls `model.forward()`. Both single-sequence and batch generation work without changes.

## Memory Considerations

This is a 30B parameter model (~61GB in bf16). Options:
- **Full GPU**: needs ~61GB VRAM (A100 80GB, etc.)
- **CPU offload**: slow but works on any machine
- **Quantization**: not covered here, but worth exploring later

## Key Differences Summary

| Aspect | Qwen3 Dense | Qwen3 MoE |
|---|---|---|
| FFN per layer | 1 large SwiGLU (3072 hidden) | 128 small SwiGLU (768 hidden each) |
| Active params/token | All | ~3B of 30B |
| New modules | — | MoEGate, SparseMoEBlock |
| Layers | 28 | 48 |
| Attention | GQA (16q/8kv) | GQA (32q/4kv) |
| Weight count per layer | 3 FFN matrices | 1 gate + 128×3 expert matrices |
| `tie_word_embeddings` | true | false |

## References

- [Qwen3 Dense Implementation](../qwen3/) — base code this builds on (reuses RMSNorm, RoPE, GQA, tokenizer, generation loop)
- [Qwen3 Technical Report](https://qwenlm.github.io/blog/qwen3/) — architecture details
- [HuggingFace Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) — model weights and config
- [HuggingFace transformers Qwen3-MoE](https://github.com/huggingface/transformers/tree/main/src/transformers/models/qwen3_moe) — reference implementation
- [Switch Transformer](https://arxiv.org/abs/2101.03961) — foundational MoE paper

## Dependencies

Managed by [uv](https://docs.astral.sh/uv/). Dependencies are declared in the root `pyproject.toml` and installed automatically on `uv run`.
