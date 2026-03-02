# GPT-OSS Inference from Scratch

A from-scratch PyTorch implementation of [GPT-OSS-20B](https://huggingface.co/openai/gpt-oss-20b) inference. Builds on the same patterns as [Qwen3-MoE](../qwen3_moe/README.md) but introduces several novel architectural ideas from OpenAI.

Target model: [GPT-OSS-20B](https://huggingface.co/openai/gpt-oss-20b) — 21B total params, ~3.6B active per token.

## Quick Start

All commands run from the **project root** (`llm-from-scratch/`):

```bash
# 1. Download model checkpoint
bash scripts/download.sh openai/gpt-oss-20b

# 2. Run inference
uv run python -m gpt_oss.main

# Custom prompt
uv run python -m gpt_oss.main -p "Explain attention sinks"
```

## What's Different from Qwen3-MoE

Both are MoE models with RMSNorm + GQA + RoPE + SwiGLU experts. Here are the **7 key differences**:

```
Qwen3-MoE                              GPT-OSS
──────────                              ───────
Standard RoPE                           YaRN RoPE (131K context)       ← Step 3
QK-Norm (RMSNorm per head)              No QK-Norm                     ← simpler
Separate Q, K, V projections            Separate Q, K, V projections   ← Step 4
No attention sinks                      Learnable attention sinks      ← Step 4
Full attention everywhere               Alternating sliding/full       ← Step 4
SwiGLU: silu(gate) * up                 Clamped SwiGLU: silu(gate) * (up+1)  ← Step 5
All linear layers bias=False            Experts + QKV have bias        ← Steps 4,5
Gate bias=False                         Router has bias                ← Step 6
128 experts, top-8                      32 experts, top-4              ← Step 6
Softmax after top-k                     Softmax after top-k            (same)
HuggingFace tokenizers                  tiktoken (o200k_base)          ← Step 2
Jinja2 chat template                    Harmony format (no template)   ← Step 2
bf16 safetensors                        MXFP4 quantized safetensors    ← Step 8
```

## Architecture Overview

```
Token IDs → Embedding → [24 × Transformer Block] → RMSNorm → LM Head → Logits
                              │
                              ├── RMSNorm
                              ├── Attention (Q/K/V + sinks + sliding window) + Residual
                              ├── RMSNorm
                              └── MoE MLP (router + 32 experts) + Residual
```

Layer types alternate: even layers use sliding-window attention (128 tokens), odd layers use full attention.

## GPT-OSS-20B Config

| Param | GPT-OSS-20B | vs Qwen3-MoE |
|---|---|---|
| `hidden_size` | 2880 | 2048 |
| `num_layers` | 24 | 48 |
| `num_attention_heads` | 64 | 32 |
| `num_kv_heads` | 8 | 4 |
| `head_dim` | 64 | 128 |
| `num_experts` | 32 | 128 |
| `experts_per_token` | 4 | 8 |
| `intermediate_size` | 2880 | 768 |
| `vocab_size` | 201088 | 151936 |
| `context_length` | 131072 | 40960 |
| `rope_theta` | 150,000 | 1,000,000 |
| `sliding_window` | 128 | — |
| `swiglu_limit` | 7.0 | — |
| `attention_bias` | true | false |
| `tie_word_embeddings` | false | false |

## File Structure

```
gpt_oss/
├── config.py       # Step 1: config with GPT-OSS fields
├── tokenizer.py    # Step 2: tiktoken wrapper
├── layers.py       # Steps 3-6: YaRN RoPE, Attention+Sinks, Clamped SwiGLU, MoE
├── model.py        # Step 7: full model
├── weights.py      # Step 8: weight mapping + MXFP4 decompression
├── generate.py     # Step 9: reuse generation loop
└── main.py         # CLI entry point
```

## Implementation Guide

Assumes familiarity with [Qwen3-MoE](../qwen3_moe/README.md). Each step focuses only on what's new.

### Step 1: Config (`config.py`)

Map `config.json` to a dataclass. New fields vs Qwen3-MoE:

```python
sliding_window: int        # 128 — local attention window for even layers
swiglu_limit: float        # 7.0 — clamp threshold for SwiGLU activation
initial_context_length: int # 4096 — original training length (for YaRN)
rope_scaling_factor: float # 32.0 — YaRN extension factor
```

Fields that disappear (vs Qwen3-MoE): `moe_norm_topk_prob` (always softmax-after-topk), `moe_hidden_dim` (equals `intermediate_size`).

### Step 2: Tokenizer (`tokenizer.py`)

GPT-OSS uses `tiktoken` (OpenAI's tokenizer), not HuggingFace tokenizers.

```python
import tiktoken

# Build from o200k_base + custom special tokens
base = tiktoken.get_encoding("o200k_base")
tokenizer = tiktoken.Encoding(
    name="o200k_harmony",
    pat_str=base._pat_str,
    mergeable_ranks=base._mergeable_ranks,
    special_tokens={**base._special_tokens, "<|startoftext|>": 199998, ...},
)
```

Key tokens: `<|startoftext|>` (199998, BOS), `<|endoftext|>` (199999, PAD), `<|return|>` (200002, EOS).

No Jinja2 chat template — GPT-OSS uses the proprietary "Harmony" format via `openai_harmony` library. We skip chat formatting and do raw text completion.

### Step 3: YaRN RoPE (`layers.py`) — NEW

Qwen3 uses standard RoPE. GPT-OSS uses [YaRN](https://arxiv.org/abs/2309.00071) (Yet another RoPE extensioN) to extend context from 4K → 131K.

The core idea: different frequency dimensions need different treatment when extending context. Low-frequency dimensions (capturing long-range patterns) need interpolation, while high-frequency dimensions (capturing local patterns) can extrapolate.

```
Standard RoPE:  inv_freq[i] = 1 / (theta ^ (2i/d))
YaRN:           inv_freq[i] = blend(interpolated[i], extrapolated[i], ramp[i])
                where ramp smoothly transitions from interpolation to extrapolation
```

YaRN also multiplies cos/sin by a learned `concentration` factor:
```
concentration = 0.1 * ln(scaling_factor) + 1.0
cos_scaled = cos * concentration
sin_scaled = sin * concentration
```

Another difference from Qwen3 RoPE: GPT-OSS applies rotation as `[x1*cos - x2*sin, x2*cos + x1*sin]` (split halves), while Qwen3 uses `[-x2, x1]` interleaved rotation. Both are valid RoPE formulations.

### Step 4: Attention with Sinks + Sliding Window (`layers.py`) — NEW

Three differences from Qwen3-MoE attention:

**4a. Separate Q, K, V projections (with bias):**
```python
q = q_proj(x)  # (batch, seq, n_heads * head_dim)
k = k_proj(x)  # (batch, seq, n_kv * head_dim)
v = v_proj(x)  # (batch, seq, n_kv * head_dim)
```

Same structure as Qwen3, but with `bias=True`. Must use separate projections (not fused) to match HF's bf16 numerics exactly — see [Notes](#notes).

**4b. Attention sinks:**
A learnable scalar per attention head that absorbs "excess" attention probability. Without sinks, tokens are forced to attend somewhere even when nothing is relevant, often dumping probability on the first token. Sinks give attention a proper "none of the above" option.

```
Standard:    softmax(Q @ K^T / sqrt(d))              → weights sum to 1 across K positions
With sinks:  softmax([Q @ K^T / sqrt(d) | sink])     → weights sum to 1 across K positions + 1 sink
             then drop the sink column before applying to V
```

The sink is a learnable parameter per head, shape `(n_heads,)`, broadcast to match the attention score dimensions.

**4c. Sliding window:**
Even-indexed layers restrict attention to the nearest 128 tokens (plus the sink). Odd-indexed layers use full causal attention. This saves memory on long sequences while maintaining some global context through alternation.

```python
sliding_window = config.sliding_window if layer_idx % 2 == 0 else 0
# Adds extra -inf masking for positions beyond the window
```

**4d. No QK-Norm:**
Qwen3 applies RMSNorm to Q and K per head. GPT-OSS skips this entirely.

**4e. Attention bias:**
GPT-OSS QKV and output projections use `bias=True`. Qwen3 uses `bias=False`.

### Step 5: Clamped SwiGLU (`layers.py`) — NEW

GPT-OSS uses a modified SwiGLU activation with clamping and a +1 bias:

```python
# Qwen3 standard SwiGLU:
output = silu(gate_proj(x)) * up_proj(x)
output = down_proj(output)

# GPT-OSS clamped SwiGLU:
# gate_up is a fused projection producing interleaved gate/up values
gate, up = gate_up[..., ::2], gate_up[..., 1::2]  # deinterleave
gate = clamp(gate, max=7.0)       # prevent explosion
up = clamp(up, min=-7.0, max=7.0) # symmetric clamp
glu = gate * sigmoid(1.702 * gate) # SiLU-like with alpha=1.702
output = glu * (up + 1)            # the +1 is the key difference
```

The `+1` bias on the linear path means the default passthrough is 1x (identity-like), and the network learns deviations. The clamping at 7.0 prevents numerical issues in lower-precision formats.

Also: GPT-OSS experts have **bias terms** on both mlp1 and mlp2. Qwen3 experts are bias-free.

### Step 6: MoE Router (`layers.py`)

Similar to Qwen3-MoE's gate, but:
- **32 experts** (not 128), **top-4** (not top-8)
- Router linear has **bias** (`nn.Linear(hidden_size, num_experts, bias=True)`)
- Softmax after top-k (same as Qwen3 with `moe_norm_topk_prob=True`)

The overall SparseMoEBlock logic is the same loop-over-experts approach.

### Step 7: Model Assembly (`model.py`)

Same pattern as Qwen3-MoE:
```
tok_emb → [24 × TransformerBlock] → final_norm → lm_head
```

But each TransformerBlock includes the attention sinks and alternating sliding/full window behavior. `tie_word_embeddings=false`.

### Step 8: Weight Loading (`weights.py`) — MXFP4

This is the biggest departure from Qwen3. GPT-OSS weights on HuggingFace are **MXFP4 quantized** (4-bit floating point with shared exponents).

Modules NOT quantized (stored as bf16): attention layers, router, embeddings, lm_head.

Modules quantized (MXFP4): all expert MLP weights.

MXFP4 decompression:
```
1. Read packed bytes (two 4-bit values per byte)
2. Extract nibbles → lookup in FP4 value table
3. Apply shared scale factors (one per block of values)
4. Result: full bf16 tensor
```

Weight name mapping (HuggingFace → our model):
```
model.embed_tokens.weight           → tok_emb.weight
model.norm.weight                   → final_norm.weight
lm_head.weight                      → lm_head.weight

# Per layer:
model.layers.{i}.input_layernorm.weight              → layers.{i}.norm1.weight
model.layers.{i}.post_attention_layernorm.weight      → layers.{i}.norm2.weight
model.layers.{i}.self_attn.q_proj.weight/bias           → layers.{i}.attn.q_proj.weight/bias
model.layers.{i}.self_attn.k_proj.weight/bias           → layers.{i}.attn.k_proj.weight/bias
model.layers.{i}.self_attn.v_proj.weight/bias           → layers.{i}.attn.v_proj.weight/bias
model.layers.{i}.self_attn.o_proj.weight/bias           → layers.{i}.attn.out.weight/bias
model.layers.{i}.self_attn.sinks                       → layers.{i}.attn.sinks
model.layers.{i}.mlp.gate.weight/bias                  → layers.{i}.moe.router.weight/bias
model.layers.{i}.mlp.mlp1_weight                       → layers.{i}.moe.mlp1_weight
model.layers.{i}.mlp.mlp1_bias                         → layers.{i}.moe.mlp1_bias
model.layers.{i}.mlp.mlp2_weight                       → layers.{i}.moe.mlp2_weight
model.layers.{i}.mlp.mlp2_bias                         → layers.{i}.moe.mlp2_bias
```

Note: expert weights are stored as 3D tensors `(num_experts, ...)` rather than individual per-expert parameters. This matches the batched einsum approach in the forward pass.

### Step 9: Generation (`generate.py`)

Reuse from Qwen3-MoE as-is. The generation loop is model-agnostic — it just calls `model.forward()` and samples from logits. Both single-sequence and batch generation work without changes.

## Key Concepts Explained

### Why Attention Sinks?

In standard attention, softmax forces probabilities to sum to 1. When a token has no particularly relevant context, it still must distribute all its probability mass somewhere. This often lands on the first token (position 0), creating an artificial "attention sink" that distorts the first token's representation.

GPT-OSS makes this explicit with a learnable sink parameter. Now attention can route probability to the sink without corrupting any token's representation. After softmax, the sink column is simply discarded before computing the weighted sum over V.

### Why YaRN over Standard RoPE?

Standard RoPE degrades when used beyond training length. Simple position interpolation (dividing all positions by a factor) works but loses resolution. YaRN is smarter:

- **High-frequency dims** (local patterns): can extrapolate naturally → keep original frequencies
- **Low-frequency dims** (long-range patterns): need interpolation → divide by scaling factor
- **Mid-frequency dims**: smooth blend between the two strategies

This preserves short-range attention precision while enabling long-range capability.

### Why Clamped SwiGLU with +1?

The `+1` bias in `glu * (up + 1)` means the default gate output is `glu * 1 = glu` — the linear path acts as identity by default. The network only needs to learn small deviations from this baseline, which stabilizes training.

Clamping at 7.0 prevents the SiLU activation from producing extremely large values, which is especially important when using low-precision formats like MXFP4.

## Memory Considerations

GPT-OSS-20B weights are stored in MXFP4 (~4 bits per expert weight), fitting in ~16GB. When decompressed to bf16 for computation, the full model is ~42GB. Options:
- **MXFP4 on GPU**: ~16GB VRAM (decompress on-the-fly per layer)
- **bf16 on GPU**: ~42GB VRAM
- **CPU**: slow but works on any machine

## Notes

Bugs fixed during implementation. The accuracy test (`test_accuracy.py`) compares greedy decoding output token-for-token against HuggingFace transformers, requiring exact string match and logprob tolerance < 0.02.

### 1. YaRN scaling applied to angles instead of sin/cos

`sin(c * angle)` is not the same as `c * sin(angle)`. The YaRN attention scaling factor must multiply the final sin/cos values, not the angles before computing sin/cos.

```python
# Before — wrong: scaling changes the frequency
sin = torch.sin(angles * attention_scaling)
cos = torch.cos(angles * attention_scaling)

# After — correct: scaling is an amplitude factor on the result
sin = torch.sin(angles) * attention_scaling
cos = torch.cos(angles) * attention_scaling
```

### 2. YaRN ramp in wavelength space vs dimension-index space

The interpolation/extrapolation blend ramp must operate over dimension indices (0, 1, 2, ..., half_dim-1), not wavelengths. HF uses `find_correction_dim` to convert rotation counts to dimension indices, then builds a linear ramp in that space.

```python
# Before — wrong: ramp in wavelength space
wavelengths = 2 * math.pi * pos_freqs
ramp = torch.clamp((wavelengths - low_wave) / (high_wave - low_wave), 0, 1)

# After — correct: ramp in dimension-index space
def find_correction_dim(num_rotations):
    return (dim * math.log(original_ctx / (num_rotations * 2 * math.pi))) / (2 * math.log(rope_base))
low = find_correction_dim(beta_fast)
high = find_correction_dim(beta_slow)
ramp = torch.clamp((torch.arange(half_dim).float() - low) / (high - low), 0, 1)
```

### 3. SwiGLU gate clamped both min and max

HF only clamps the gate's upper bound. Clamping min on the gate blocks negative values, which changes the activation shape.

```python
# Before — wrong: clamps gate to [0, 7], killing negative inputs
x_gate = x_gate.clamp(min=0, max=7)

# After — correct: only prevent overflow on the positive side
x_gate = x_gate.clamp(max=7)
```

### 4. Missing causal mask during prefill

During prefill (seq_len > 1), a causal mask is needed. Without it, tokens attend to future positions.

```python
# Before — no causal mask, only sliding window mask

# After — add causal mask when processing multiple tokens
if q_len > 1 and attn_mask is None:
    kv_offset = kv_len - q_len
    causal = torch.triu(torch.ones(q_len, kv_len, device=x.device), diagonal=kv_offset + 1).bool()
    scores.masked_fill_(causal, float("-inf"))
```

### 5. RMSNorm epsilon not propagated from config

The model config specifies `rms_norm_eps=1e-5`, but all norms used the default `1e-6`. The wrong epsilon changes the normalization output.

```python
# Before — hardcoded default
self.norm1 = RMSNorm(emb_dim)  # uses eps=1e-6

# After — propagate from config
self.norm1 = RMSNorm(emb_dim, eps=rms_norm_eps)  # uses eps=1e-5
```

### 6. RoPE sin/cos not cast to input dtype

RoPE buffers are computed in float32 during `_build_buffers()`. When applied to bf16 Q/K tensors, PyTorch promotes Q/K to float32 for the multiply, producing different results than HF which casts sin/cos to bf16 first.

```python
# Before — float32 buffers promote Q, K to float32
sin = self.sin[position_ids]  # float32
cos = self.cos[position_ids]  # float32
output = x * cos + rotated * sin  # x gets promoted to float32

# After — cast to match input dtype, keeping computation in bf16
sin = self.sin[position_ids].to(x.dtype)  # bf16
cos = self.cos[position_ids].to(x.dtype)  # bf16
output = x * cos + rotated * sin  # stays in bf16
```

### 7. Missing explicit max subtraction before softmax

PyTorch's `F.softmax` does max subtraction internally in float32, but HF subtracts in bf16 before calling softmax. The bf16 subtraction rounds differently, producing different attention weights.

```python
# Before — rely on softmax's internal max-sub (float32)
attn = F.softmax(scores, dim=-1)

# After — explicit max-sub in bf16, matching HF
scores = scores - scores.max(dim=-1, keepdim=True).values
attn = F.softmax(scores, dim=-1, dtype=scores.dtype)
```

### 8. `F.linear` vs `.T.contiguous()` in MoE experts — accuracy vs performance tradeoff

HF uses separate `matmul + bias` (not fused `torch.addmm`), and `.T.contiguous()` to force a specific BLAS kernel path. Matching this exactly gives token-for-token accuracy but is **extremely slow**:

`.T.contiguous()` copies the entire weight matrix every forward pass. With 4 active experts × 2 projections × 24 layers, that's **~4.8 GB of memcpy per decode step** — single-threaded memory copies that prevent multi-core CPU utilization (~110% CPU instead of ~400%).

```python
# Accurate (matches HF token-for-token) but slow:
# ~4.8 GB of weight copies per decode step, single-threaded memcpy bottleneck
expert_output = tokens @ weight[expert_idx].T.contiguous() + bias[expert_idx]

# Fast (uses multi-threaded BLAS, no copies) but may differ in bf16 rounding:
# F.linear handles transpose via BLAS trans flag — zero-copy
expert_output = F.linear(tokens, weight[expert_idx], bias[expert_idx])
```

We use `F.linear` for performance. The bf16 rounding difference is negligible for practical use but will not pass the strict token-for-token accuracy test against HF.

### 9. Attention sink concatenated at beginning vs end

HF concatenates the sink at the **end** of attention scores: `cat([scores, sink])`. We had it at the beginning: `cat([sink, scores])`. While softmax is mathematically permutation-invariant, bf16 accumulates left-to-right, producing different rounding.

```python
# Before — sink at position 0, strip first
scores = torch.concat([sinks, scores], dim=-1)
attn = F.softmax(scores, dim=-1)
attn = attn[:, :, :, 1:]  # drop sink

# After — sink at end, strip last, matching HF accumulation order
scores = torch.concat([scores, sinks], dim=-1)
attn = F.softmax(scores, dim=-1)
attn = attn[:, :, :, :-1]  # drop sink
```

### 10. Fused QKV projection vs separate Q, K, V

A single fused matmul `x @ [Wq; Wk; Wv]^T` (5120 output cols) hits different BLAS tiling than three separate matmuls `x @ Wq^T` (4096), `x @ Wk^T` (512), `x @ Wv^T` (512). Different tiling means different bf16 accumulation order per dot product.

```python
# Before — one large matmul, then split
qkv = self.qkv_proj(x)  # (batch, seq, 5120)
Q, K, V = torch.split(qkv, [n_heads*d, n_kv*d, n_kv*d], dim=2)

# After — three separate projections, matching HF matrix sizes
Q = self.q_proj(x)  # (batch, seq, 4096)
K = self.k_proj(x)  # (batch, seq, 512)
V = self.v_proj(x)  # (batch, seq, 512)
```

## References

- [OpenAI GPT-OSS GitHub](https://github.com/openai/gpt-oss) — official implementation
- [GPT-OSS Model Card](https://arxiv.org/abs/2508.10925) — architecture paper
- [HuggingFace GPT-OSS-20B](https://huggingface.co/openai/gpt-oss-20b) — model weights
- [YaRN Paper](https://arxiv.org/abs/2309.00071) — RoPE extension method
- [Attention Sinks Paper](https://arxiv.org/abs/2309.17453) — StreamingLLM / attention sink concept
- [Qwen3-MoE Implementation](../qwen3_moe/) — base patterns this builds on
