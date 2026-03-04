# Qwen3-MoE Inference from Scratch

A from-scratch PyTorch implementation of Qwen3-MoE (Mixture of Experts) inference. Builds on the [Qwen3 dense implementation](../qwen3/README.md) — same attention, tokenizer, and generation loop, but replaces the dense FFN with sparse expert routing.

Target model: [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B)

No `transformers` library — just raw tensor operations. Read [blog post](https://blog.yellowday.day/posts/qwen3_moe_from_scratch/) for details behind.

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
| `-m`, `--model` | `Qwen3-30B-A3B`, `Qwen3-235B-A22B` | `Qwen3-30B-A3B` |
| `-p`, `--prompt` | User prompt text | `Which is bigger, 9.9 or 9.11?` |
| `-t`, `--temperature` | Sampling temperature (0 = greedy) | `1.0` |
| `-k`, `--top-k` | Top-k filtering (-1 = disabled) | `-1` |
| `-n`, `--max-tokens` | Max new tokens to generate | `1024` |
| `--thinking` | Enable thinking mode | off |
| `-d`, `--device` | `cuda`, `cpu`, or `auto` | `auto` |

If a checkpoint is missing, you'll be prompted to download it automatically (requires `huggingface_hub`).

## Architecture Overview

```
Token IDs → Embedding → [48 × MoE Transformer Block] → RMSNorm → LM Head → Logits → Sample → Token
                              │
                              ├── RMSNorm
                              ├── GQA (Grouped Query Attention) + Residual
                              ├── RMSNorm
                              └── SparseMoEBlock + Residual
                                    │
                                    ├── MoERouter → softmax → top-8 of 128 experts
                                    └── Fused 3D expert weights (gate_up_proj, down_proj)
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
┌─── MoERouter ────────┐
│ Linear(2048, 128)     ← produces 128 scores, one per expert
│ Softmax over all 128  ← normalize across ALL experts first
│ Top-8 selection       ← then pick 8 highest
│ Re-normalize top-8    ← weights sum to 1 (norm_topk_prob=True)
└────────┬─────────────┘
         │  routing_weights: (num_tokens, 8)
         │  selected_experts: (num_tokens, 8)
         ▼
┌─ For each active expert ──┐
│  Find tokens routed here   │
│  Expert_i(tokens) × weight │  ← fused gate+up projection, SiLU, down projection
└─────────┬──────────────────┘
          │
          ▼  Sum all expert outputs per token
     Output (2048-dim)
```

Expert weights are stored as fused 3D tensors (matches HF transformers 5.x `Qwen3MoeExperts`):
```
gate_up_proj: (128, 2×768, 2048)  ← gate and up projections fused per expert
down_proj:    (128, 2048, 768)

Expert_i(x):
  gate, up = linear(x, gate_up_proj[i]).chunk(2)   ← single fused matmul
  output = linear(SiLU(gate) * up, down_proj[i])
```

Why it works: 128 experts × 768 intermediate = 98,304 total capacity, but only 8 × 768 = 6,144 active per token. The model learns to specialize experts for different token types.

## File Structure

```
qwen3_moe/
├── config.py       # Step 1: config dataclass with MoE fields
├── tokenizer.py    # (reused from qwen3, identical)
├── layers.py       # Steps 2-4: all layers (RMSNorm, RoPE, GQA, MoERouter, SparseMoEBlock, MoETransformersBlock)
├── model.py        # Step 5: full model assembly
├── weights.py      # Step 6: weight mapping for router + experts
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

Copy or import these unchanged from Qwen3: `RMSNorm`, `RoPE`, `GroupQueryAttention`.

No separate `SwiGLUFFN` class is needed — expert computations use fused 3D weight tensors directly (see Step 4).

### Step 3: MoE Router (`layers.py`)

The router decides which experts process each token. This is the key new component.

```python
class MoERouter(nn.Module):
    # proj: Linear(emb_dim, n_experts, bias=False)
    # i.e., Linear(2048, 128)
```

Forward logic:
1. **Score all experts**: `scores = proj(hidden_states)` → shape `(num_tokens, 128)`
2. **Softmax over all experts**: `softmax(scores)` → probabilities across all 128 experts
3. **Top-k selection**: `torch.topk(probs, n_experts_per_token)` → top-8 values and indices
4. **Re-normalize** (when `moe_norm_topk_prob=True`): divide by sum so the 8 weights sum to 1.0

Key design choice: softmax is applied **before** top-k selection (softmax-before-topk). This matches the HF transformers 5.x computation order — all 128 experts compete in the softmax, then the top-8 are selected.

Returns: `routing_weights (num_tokens, 8)` and `selected_experts (num_tokens, 8)`

### Step 4: SparseMoEBlock (`layers.py`)

Combines the router with fused 3D expert weight tensors. This replaces the single dense FFN.

```python
class SparseMoEBlock(nn.Module):
    # router: MoERouter
    # gate_up_proj: Parameter(n_experts, 2 * moe_hidden_dim, emb_dim)  ← fused 3D tensor
    # down_proj:    Parameter(n_experts, emb_dim, moe_hidden_dim)      ← fused 3D tensor
```

Forward logic — the **loop-over-active-experts approach**:
```
1. Flatten input: (batch, seq, hidden) → (num_tokens, hidden)
2. Get routing: weights, expert_indices = router(hidden_states)
3. Allocate output: zeros_like(hidden_states)
4. For each expert_idx in selected_experts.unique():
     - torch.where(selected_experts == expert_idx) → find routed tokens
     - Fused gate+up: linear(tokens, gate_up_proj[expert_idx]).chunk(2)
     - SwiGLU: SiLU(gate) * up
     - Down: linear(result, down_proj[expert_idx])
     - Scale by routing weight and accumulate via index_add_
5. Reshape output back to (batch, seq, hidden)
```

Why loop over experts (not tokens)? Each expert processes a variable-sized batch of tokens. Looping over experts gives larger matrix multiplies than looping over tokens (where each runs 8 tiny matmuls). Using `selected_experts.unique()` skips experts that received no tokens.

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
# MoE router (one per layer, mapped directly):
model.layers.{i}.mlp.gate.weight → layers.{i}.moe_ffn.router.proj.weight

# MoE experts (128 per layer, 3 matrices each — fused during loading):
model.layers.{i}.mlp.experts.{j}.gate_proj.weight ─┐
model.layers.{i}.mlp.experts.{j}.up_proj.weight   ─┼→ layers.{i}.moe_ffn.gate_up_proj  (n_experts, 2*moe_hidden_dim, emb_dim)
model.layers.{i}.mlp.experts.{j}.down_proj.weight  ─→ layers.{i}.moe_ffn.down_proj     (n_experts, emb_dim, moe_hidden_dim)
```

Expert weight loading is a two-phase process:
1. **Buffer**: collect per-expert `gate_proj`, `up_proj`, `down_proj` tensors from safetensors shards (matched by regex)
2. **Fuse**: for each layer, `torch.cat` gate+up per expert, then `torch.stack` across all 128 experts into the 3D `gate_up_proj` tensor. `down_proj` is stacked directly.

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
| FFN per layer | 1 large SwiGLU (3072 hidden) | 128 small experts (768 hidden each), fused 3D weights |
| Active params/token | All | ~3B of 30B |
| New modules | — | MoERouter, SparseMoEBlock |
| Layers | 28 | 48 |
| Attention | GQA (16q/8kv) | GQA (32q/4kv) |
| Expert weights | 3 separate FFN matrices | fused `gate_up_proj` (n_experts, 2\*hidden, emb) + `down_proj` (n_experts, emb, hidden) |
| Routing | — | softmax-before-topk (matches HF 5.x) |
| `tie_word_embeddings` | true | false |

## Tests

Run from the project root. Tests require downloaded checkpoints and `--model` flag.

```bash
# Knowledge tests: math, facts, thinking mode, batch correctness
uv run python -m pytest tests/qwen3_moe/test_knowledge.py -m slow -v --model Qwen3-30B-A3B -s

# Accuracy tests: token-level match vs HuggingFace transformers
uv run python -m pytest tests/qwen3_moe/test_accuracy.py -m slow -v --model Qwen3-30B-A3B -s
```

Notes:
- Batch generation doesn't guarantee exact token match with single-sequence generation because the router's top-k expert selection is discontinuous — tiny floating-point differences from padding masks can flip which experts are selected. The batch test checks semantic correctness instead.
- The accuracy test forces HF to use loop-based expert forward (`_experts_implementation = None`) instead of the default `grouped_mm` kernel, which produces different floating-point results. Our loop-based scratch implementation matches the loop-based HF path exactly.

## TODO

- [ ] Implement and test alternative expert dispatch strategies (e.g. `grouped_mm`)

## References

- [Qwen3 Dense Implementation](../qwen3/) — base code this builds on (reuses RMSNorm, RoPE, GQA, tokenizer, generation loop)
- [Qwen3 Technical Report](https://qwenlm.github.io/blog/qwen3/) — architecture details
- [HuggingFace Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) — model weights and config
- [HuggingFace transformers Qwen3-MoE](https://github.com/huggingface/transformers/tree/main/src/transformers/models/qwen3_moe) — reference implementation
- [Switch Transformer](https://arxiv.org/abs/2101.03961) — foundational MoE paper

## Dependencies

Managed by [uv](https://docs.astral.sh/uv/). Dependencies are declared in the root `pyproject.toml` and installed automatically on `uv run`.

**Note:** The accuracy tests require `transformers>=5.2`. Our implementation uses fused 3D expert weight tensors (`gate_up_proj`, `down_proj`) and softmax-before-topk routing to match the HuggingFace transformers 5.x computation order exactly.
