# Tensor Parallelism Implementation Guide

Step-by-step guide to add tensor parallelism for **Qwen3**, **Qwen3-MoE**, and **GPT-OSS** models.

## Background

### What We Have

```
parallel/
  comm.py          # init_process_group, get_rank, get_world_size, all_reduce
  tensor.py        # ColumnParallelLinear, RowParallelLinear
  __init__.py
```

- **ColumnParallelLinear**: splits OUTPUT dim across ranks. Weight: `(out_features // world_size, in_features)`. No communication in forward.
- **RowParallelLinear**: splits INPUT dim across ranks. Weight: `(out_features, in_features // world_size)`. `all_reduce` in forward.

### Core Principle

Tensor parallelism always pairs **Column -> Row**:
- Column splits the output (each rank computes a slice)
- Row recombines via all_reduce (each rank gets the full result)

This means: **one all_reduce per Column-Row pair** (not per layer).

```
Input (replicated on all ranks)
  |
  ColumnParallelLinear   (no comm)      -- output is SPLIT across ranks
  |
  element-wise ops       (no comm)      -- each rank works on its slice
  |
  RowParallelLinear      (all_reduce)   -- output is FULL on all ranks
  |
Output (replicated on all ranks)
```

---

## Step 1: Add VocabParallelEmbedding to `parallel/tensor.py`

The embedding table can be huge (vocab_size * emb_dim). Split vocab across ranks.

```python
class VocabParallelEmbedding(nn.Module):
    """Embedding with vocab split across ranks.

    Each rank stores vocab_size // world_size rows.
    Tokens outside this rank's range produce zeros; all_reduce combines.
    """
    def __init__(self, vocab_size, emb_dim, world_size=1):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.world_size = world_size
        self.vocab_per_rank = vocab_size // world_size
        self.rank = get_rank()
        self.vocab_start = self.rank * self.vocab_per_rank
        self.vocab_end = self.vocab_start + self.vocab_per_rank
        self.weight = nn.Parameter(torch.empty(self.vocab_per_rank, emb_dim))

    def forward(self, token_ids):
        mask = (token_ids >= self.vocab_start) & (token_ids < self.vocab_end)
        local_ids = (token_ids - self.vocab_start).clamp(0, self.vocab_per_rank - 1)
        output = F.embedding(local_ids, self.weight)
        output = output * mask.unsqueeze(-1)  # zero out tokens not on this rank
        if self.world_size > 1:
            all_reduce(output)
        return output
```

Export it from `parallel/__init__.py`.

---

## Step 2: Qwen3 Parallel Model

### 2.1 Create `qwen3/parallel_layers.py`

The key insight: identify every `nn.Linear` and decide Column vs Row.

#### Attention (GroupQueryAttention)

```
x (replicated)
  |
  q_proj: ColumnParallel  (emb_dim -> n_heads * head_dim)     -- split heads across ranks
  k_proj: ColumnParallel  (emb_dim -> n_kv_groups * head_dim) -- split kv groups across ranks
  v_proj: ColumnParallel  (emb_dim -> n_kv_groups * head_dim) -- split kv groups across ranks
  |
  [rope, attention computation -- each rank works on its head slice]
  |
  o_proj: RowParallel      (n_heads * head_dim -> emb_dim)    -- all_reduce to recombine
  |
x (replicated)
```

What changes:
- `n_heads` becomes `n_heads // world_size` on each rank
- `n_kv_groups` becomes `n_kv_groups // world_size` on each rank
- `q_norm`, `k_norm` stay the same (they operate per-head, head_dim unchanged)
- The `.view()` reshape uses local head counts

```python
class ParallelGroupQueryAttention(nn.Module):
    def __init__(self, emb_dim, n_heads, n_kv_groups, head_dim, world_size=1):
        super().__init__()
        self.emb_dim = emb_dim
        self.n_heads = n_heads // world_size        # LOCAL head count
        self.n_kv_groups = n_kv_groups // world_size  # LOCAL kv group count
        self.head_dim = head_dim
        self.world_size = world_size

        self.q_proj = ColumnParallelLinear(emb_dim, n_heads * head_dim, world_size, bias=False)
        self.k_proj = ColumnParallelLinear(emb_dim, n_kv_groups * head_dim, world_size, bias=False)
        self.v_proj = ColumnParallelLinear(emb_dim, n_kv_groups * head_dim, world_size, bias=False)
        self.o_proj = RowParallelLinear(n_heads * head_dim, emb_dim, world_size, bias=False)

        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)

    def forward(self, x, rope, position_ids, kv_cache=None, attn_mask=None):
        batch, seq_len, _ = x.shape

        # These use LOCAL head counts in reshape
        Q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch, seq_len, self.n_kv_groups, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch, seq_len, self.n_kv_groups, self.head_dim).transpose(1, 2)

        # q_norm, k_norm, rope, kv_cache, attention -- same logic, just fewer heads per rank
        # ... (same as original, using self.n_heads and self.n_kv_groups)

        context = context.transpose(1, 2).reshape(batch, seq_len, -1)
        output = self.o_proj(context)  # all_reduce happens inside
        return output, new_kv_cache
```

**Constraint**: `n_heads % world_size == 0` and `n_kv_groups % world_size == 0`.

#### FFN (SwiGLUFFN)

```
x (replicated)
  |
  gate_proj: ColumnParallel  (emb_dim -> hidden_dim)  -- split hidden across ranks
  up_proj:   ColumnParallel  (emb_dim -> hidden_dim)  -- split hidden across ranks
  |
  silu(gate) * up  -- element-wise on split hidden
  |
  down_proj: RowParallel     (hidden_dim -> emb_dim)  -- all_reduce to recombine
  |
x (replicated)
```

```python
class ParallelSwiGLUFFN(nn.Module):
    def __init__(self, emb_dim, hidden_dim, world_size=1):
        super().__init__()
        self.gate_proj = ColumnParallelLinear(emb_dim, hidden_dim, world_size, bias=False)
        self.up_proj = ColumnParallelLinear(emb_dim, hidden_dim, world_size, bias=False)
        self.down_proj = RowParallelLinear(hidden_dim, emb_dim, world_size, bias=False)

    def forward(self, x):
        gate = F.silu(self.gate_proj(x))
        output = self.down_proj(gate * self.up_proj(x))
        return output
```

#### TransformerBlock

```python
class ParallelTransformerBlock(nn.Module):
    def __init__(self, config, world_size=1):
        super().__init__()
        self.norm1 = RMSNorm(config.emb_dim)
        self.attn = ParallelGroupQueryAttention(
            config.emb_dim, config.n_heads, config.n_kv_groups, config.head_dim, world_size
        )
        self.norm2 = RMSNorm(config.emb_dim)
        self.ffn = ParallelSwiGLUFFN(config.emb_dim, config.hidden_dim, world_size)

    def forward(self, x, rope, position_ids, kv_cache=None, attn_mask=None):
        # same residual pattern as original
        ...
```

### 2.2 Create `qwen3/parallel_model.py`

```python
class ParallelQwen3Model(nn.Module):
    def __init__(self, config, world_size=1):
        super().__init__()
        self.tok_emb = VocabParallelEmbedding(config.vocab_size, config.emb_dim, world_size)
        self.rope = RoPE(config.head_dim, config.rope_base, config.context_length)
        self.layers = nn.ModuleList([
            ParallelTransformerBlock(config, world_size) for _ in range(config.n_layers)
        ])
        self.final_norm = RMSNorm(config.emb_dim)
        # lm_head: ColumnParallel to split vocab output, gather later
        self.lm_head = ColumnParallelLinear(config.emb_dim, config.vocab_size, world_size, bias=False)

    def forward(self, token_ids, position_ids=None, kv_cache=None, attn_mask=None):
        # same as original
        ...
        logits = self.lm_head(x)  # (batch, seq, vocab_size // world_size)
        return logits, new_kv_caches
```

Note: `lm_head` outputs `vocab_size // world_size`. During generation, each rank computes argmax on its slice, then coordinate across ranks to find the global argmax (or gather the full logits).

### 2.3 Create `qwen3/parallel_weights.py`

Each rank loads ALL safetensors but only keeps its shard. The key change is **slicing weights** during loading.

```python
def shard_weight(tensor, dim, rank, world_size):
    """Slice tensor along dim for this rank."""
    size = tensor.shape[dim]
    shard_size = size // world_size
    start = rank * shard_size
    return tensor.narrow(dim, start, shard_size).contiguous()
```

Weight loading rules for each key:

| Layer | Shard rule |
|-------|-----------|
| `tok_emb.weight` | shard dim=0 (vocab rows) |
| `attn.q_proj.weight` | shard dim=0 (output heads) |
| `attn.k_proj.weight` | shard dim=0 (output kv groups) |
| `attn.v_proj.weight` | shard dim=0 (output kv groups) |
| `attn.o_proj.weight` | shard dim=1 (input heads) |
| `ffn.gate_proj.weight` | shard dim=0 (output hidden) |
| `ffn.up_proj.weight` | shard dim=0 (output hidden) |
| `ffn.down_proj.weight` | shard dim=1 (input hidden) |
| `lm_head.weight` | shard dim=0 (output vocab) |
| `*.norm*.weight` | NO shard (replicated) |
| `attn.q_norm.weight` | NO shard (per-head, head_dim unchanged) |
| `attn.k_norm.weight` | NO shard (per-head, head_dim unchanged) |

The general rule:
- **ColumnParallel weights**: shard dim=0 (output dim), also shard bias dim=0 if present
- **RowParallel weights**: shard dim=1 (input dim), bias is NOT sharded (replicated)
- **VocabParallelEmbedding weights**: shard dim=0 (vocab rows)
- **Everything else**: replicated (norms, RoPE buffers)

---

## Step 3: Qwen3-MoE Parallel Model

Same attention parallelism as Qwen3. The difference is the MoE FFN block.

### 3.1 MoE FFN Parallelism

Qwen3-MoE uses **fused 3D expert weights**:
- `gate_up_proj`: `(n_experts, 2 * moe_hidden_dim, emb_dim)`
- `down_proj`: `(n_experts, emb_dim, moe_hidden_dim)`

**Strategy: Tensor parallel within each expert** (split hidden_dim, keep all experts on all ranks).

```
x (replicated)
  |
  router.proj (replicated)  -- each rank needs full routing decisions
  |
  gate_up_proj: shard dim=1 (hidden_dim axis)
    shape: (n_experts, 2 * moe_hidden_dim // world_size, emb_dim)
  |
  chunk -> silu(gate) * up   -- element-wise on split hidden
  |
  down_proj: shard dim=2 (hidden_dim axis), all_reduce output
    shape: (n_experts, emb_dim, moe_hidden_dim // world_size)
  |
x (replicated, after all_reduce)
```

```python
class ParallelSparseMoEBlock(nn.Module):
    def __init__(self, emb_dim, n_experts, n_experts_per_token, moe_hidden_dim, moe_norm_topk_prob, world_size=1):
        super().__init__()
        self.world_size = world_size
        self.local_moe_hidden_dim = moe_hidden_dim // world_size

        # Router is replicated (small, needs full expert scores)
        self.router = MoERouter(emb_dim, n_experts, n_experts_per_token, moe_norm_topk_prob)

        # Expert weights sharded along hidden_dim
        self.gate_up_proj = nn.Parameter(
            torch.empty(n_experts, 2 * self.local_moe_hidden_dim, emb_dim)
        )
        self.down_proj = nn.Parameter(
            torch.empty(n_experts, emb_dim, self.local_moe_hidden_dim)
        )

    def forward(self, x):
        batch, seq_len, emb_dim = x.shape
        hidden_states = x.view(-1, emb_dim)
        routing_weights, selected_experts = self.router(hidden_states)
        final_hidden_states = torch.zeros_like(hidden_states)

        for expert_idx in selected_experts.unique():
            token_idx, top_k_pos = torch.where(selected_experts == expert_idx)
            current_state = hidden_states[token_idx]

            gate, up = F.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = F.silu(gate) * up
            current_hidden_states = F.linear(current_hidden_states, self.down_proj[expert_idx])
            # NOTE: do NOT all_reduce here per expert. Accumulate first.

            current_hidden_states *= routing_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        # all_reduce ONCE after all experts (not per expert)
        if self.world_size > 1:
            all_reduce(final_hidden_states)

        return final_hidden_states.view(batch, seq_len, emb_dim)
```

### 3.2 Weight sharding for MoE 3D tensors

| Layer | Original shape | Shard rule |
|-------|---------------|-----------|
| `moe_ffn.gate_up_proj` | `(n_experts, 2*moe_hidden_dim, emb_dim)` | shard dim=1 |
| `moe_ffn.down_proj` | `(n_experts, emb_dim, moe_hidden_dim)` | shard dim=2 |
| `moe_ffn.router.proj.weight` | `(n_experts, emb_dim)` | NO shard |

For qwen3_moe `weights.py`, the expert fusion logic must shard during fusion:
```python
# When fusing per-expert weights into 3D:
gate_up = torch.stack([
    torch.cat([parts[(j, "gate_proj")], parts[(j, "up_proj")]], dim=0)
    for j in range(n_experts)
])  # (n_experts, 2 * moe_hidden_dim, emb_dim)
gate_up = shard_weight(gate_up, dim=1, rank=rank, world_size=world_size)

down = torch.stack([
    parts[(j, "down_proj")] for j in range(n_experts)
])  # (n_experts, emb_dim, moe_hidden_dim)
down = shard_weight(down, dim=2, rank=rank, world_size=world_size)
```

---

## Step 4: GPT-OSS Parallel Model

GPT-OSS follows the same tensor parallel pattern as Qwen3-MoE, with these differences:

### 4.1 Attention differences

- **bias=True** on q/k/v/out projections: biases must also be sharded for ColumnParallel (dim=0), but NOT sharded for RowParallel (replicated, summed once via all_reduce)
- **No q_norm/k_norm**: nothing extra to handle
- **Attention sinks**: `self.sinks = nn.Parameter(shape (1, n_heads, 1, 1))` -- shard along n_heads dim (dim=1). Each rank stores `(1, n_heads // world_size, 1, 1)`
- **Sliding window**: no change needed (works per-head, same logic)

```python
class ParallelGPTOSSAttention(nn.Module):
    def __init__(self, emb_dim, n_heads, n_kv_groups, head_dim, layer_idx, sliding_window, world_size=1):
        super().__init__()
        self.n_heads = n_heads // world_size
        self.n_kv_groups = n_kv_groups // world_size
        self.head_dim = head_dim
        self.group_size = n_heads // n_kv_groups  # same ratio
        self.layer_idx = layer_idx
        self.sliding_window = sliding_window

        self.sinks = nn.Parameter(torch.empty(1, self.n_heads, 1, 1))  # local heads
        self.q_proj = ColumnParallelLinear(emb_dim, n_heads * head_dim, world_size, bias=True)
        self.k_proj = ColumnParallelLinear(emb_dim, n_kv_groups * head_dim, world_size, bias=True)
        self.v_proj = ColumnParallelLinear(emb_dim, n_kv_groups * head_dim, world_size, bias=True)
        self.out = RowParallelLinear(n_heads * head_dim, emb_dim, world_size, bias=True)

    # forward: same as original, using local n_heads/n_kv_groups
```

### 4.2 MoE FFN differences

- Expert weights have **biases**: `gate_up_proj_bias (n_experts, moe_hidden_dim*2)` and `down_proj_bias (n_experts, emb_dim)`
- `gate_up_proj_bias`: shard dim=1 (follows gate_up_proj_weight sharding)
- `down_proj_bias`: NO shard (output dim, replicated -- combined via all_reduce on the output)
- **SwiGLUFFN activation** has no parameters (just interleaved indexing on the split tensor)
- **MXFP4 decompression**: decompress first, THEN shard. Decompression in `weights.py` stays the same, just add sharding after `decompress_mxfp4()`

### 4.3 Weight sharding for GPT-OSS

| Layer | Original shape | Shard rule |
|-------|---------------|-----------|
| `attn.q_proj.weight` | `(n_heads*head_dim, emb_dim)` | shard dim=0 |
| `attn.q_proj.bias` | `(n_heads*head_dim,)` | shard dim=0 |
| `attn.k_proj.weight` | `(n_kv*head_dim, emb_dim)` | shard dim=0 |
| `attn.k_proj.bias` | `(n_kv*head_dim,)` | shard dim=0 |
| `attn.v_proj.weight` | `(n_kv*head_dim, emb_dim)` | shard dim=0 |
| `attn.v_proj.bias` | `(n_kv*head_dim,)` | shard dim=0 |
| `attn.out.weight` | `(emb_dim, n_heads*head_dim)` | shard dim=1 |
| `attn.out.bias` | `(emb_dim,)` | NO shard |
| `attn.sinks` | `(1, n_heads, 1, 1)` | shard dim=1 |
| `moe_ffn.gate_up_proj_weight` | `(n_exp, hidden*2, emb)` | shard dim=1 |
| `moe_ffn.gate_up_proj_bias` | `(n_exp, hidden*2)` | shard dim=1 |
| `moe_ffn.down_proj_weight` | `(n_exp, emb, hidden)` | shard dim=2 |
| `moe_ffn.down_proj_bias` | `(n_exp, emb)` | NO shard |
| `moe_ffn.router.proj.*` | | NO shard |
| `*.norm*.weight` | | NO shard |

---

## Step 5: Weight Loading Pattern

Create a shared utility in `parallel/weights.py`:

```python
def shard_weight(tensor, dim, rank, world_size):
    """Slice tensor along dim for this rank."""
    if world_size <= 1:
        return tensor
    size = tensor.shape[dim]
    assert size % world_size == 0, f"dim {dim} size {size} not divisible by {world_size}"
    shard_size = size // world_size
    start = rank * shard_size
    return tensor.narrow(dim, start, shard_size).contiguous()
```

Each model's `parallel_weights.py` follows this pattern:

```python
def load_weights(model, model_dir, dtype=None, rank=0, world_size=1):
    for f in sorted(model_dir.glob("*.safetensors")):
        shard = load_file(str(f))
        renamed = {}
        for hf_key, tensor in shard.items():
            new_key = rename_hf_key(hf_key)
            # Apply sharding based on key suffix
            if new_key.endswith(("q_proj.weight", "k_proj.weight", "v_proj.weight")):
                tensor = shard_weight(tensor, dim=0, rank=rank, world_size=world_size)
            elif new_key.endswith("o_proj.weight"):
                tensor = shard_weight(tensor, dim=1, rank=rank, world_size=world_size)
            elif new_key.endswith(("gate_proj.weight", "up_proj.weight")):
                tensor = shard_weight(tensor, dim=0, rank=rank, world_size=world_size)
            elif new_key.endswith("down_proj.weight"):
                tensor = shard_weight(tensor, dim=1, rank=rank, world_size=world_size)
            elif new_key.endswith("tok_emb.weight"):
                tensor = shard_weight(tensor, dim=0, rank=rank, world_size=world_size)
            elif new_key.endswith("lm_head.weight"):
                tensor = shard_weight(tensor, dim=0, rank=rank, world_size=world_size)
            # else: replicated (norms, etc.)
            renamed[new_key] = tensor
        model.load_state_dict(renamed, strict=False, assign=True)
```

---

## Step 6: Generation / Inference

In `generate.py`, the key change is handling the split `lm_head` output.

**Option A: Gather logits** (simpler, more memory)
```python
# Each rank has logits of shape (batch, seq, vocab_size // world_size)
# all_gather to get full (batch, seq, vocab_size) on every rank
local_logits = model(token_ids, ...)
all_logits = [torch.zeros_like(local_logits) for _ in range(world_size)]
dist.all_gather(all_logits, local_logits)
full_logits = torch.cat(all_logits, dim=-1)
next_token = full_logits[:, -1, :].argmax(dim=-1)
```

**Option B: Local argmax + reduce** (less memory, works for greedy)
```python
local_logits = model(token_ids, ...)[:, -1, :]  # (batch, vocab_size // world_size)
local_max_val, local_max_idx = local_logits.max(dim=-1)
# Adjust local index to global index
local_max_idx += rank * (vocab_size // world_size)
# all_reduce to find global max (need custom reduce)
```

Option A is recommended for simplicity.

---

## Step 7: Launching Multi-Process Inference

```python
# parallel_main.py
import torch.multiprocessing as mp
from parallel import init_process_group, destroy_process_group

def run_worker(rank, world_size, model_dir, prompt):
    init_process_group(rank, world_size)

    config = Config.from_model_dir(model_dir)
    with torch.device("meta"):
        model = ParallelModel(config, world_size=world_size)
    load_weights(model, model_dir, dtype=config.dtype, rank=rank, world_size=world_size)

    # generate...

    destroy_process_group()

if __name__ == "__main__":
    world_size = 2
    mp.spawn(run_worker, args=(world_size, model_dir, prompt), nprocs=world_size)
```

---

## Implementation Order

Recommended order to implement:

1. **`parallel/tensor.py`** -- add `VocabParallelEmbedding`
2. **`parallel/weights.py`** -- add `shard_weight` utility
3. **`qwen3/parallel_layers.py`** -- `ParallelGroupQueryAttention`, `ParallelSwiGLUFFN`, `ParallelTransformerBlock`
4. **`qwen3/parallel_model.py`** -- `ParallelQwen3Model`
5. **`qwen3/parallel_weights.py`** -- sharded weight loading
6. **Test qwen3 parallel** (compare output with non-parallel, world_size=1 should match exactly)
7. **`qwen3_moe/parallel_layers.py`** -- reuse attn from qwen3 parallel, add `ParallelSparseMoEBlock`
8. **`qwen3_moe/parallel_model.py`** + **`qwen3_moe/parallel_weights.py`**
9. **Test qwen3_moe parallel**
10. **`gpt_oss/parallel_layers.py`** -- handle bias, sinks, fused SwiGLU, MXFP4
11. **`gpt_oss/parallel_model.py`** + **`gpt_oss/parallel_weights.py`**
12. **Test gpt_oss parallel**

---

## Checklist Per Model

For each model, verify:

- [ ] `world_size=1` produces identical output to the non-parallel model
- [ ] `n_heads % world_size == 0` and `n_kv_groups % world_size == 0`
- [ ] `hidden_dim % world_size == 0` (and `moe_hidden_dim` for MoE)
- [ ] `vocab_size % world_size == 0`
- [ ] All-reduce count matches expectations:
  - Dense (qwen3): 2 per layer (one for attn o_proj, one for ffn down_proj) + 1 for embedding
  - MoE (qwen3_moe, gpt_oss): 2 per layer (one for attn, one for MoE block after all experts) + 1 for embedding
- [ ] Weight shapes after sharding match model parameter shapes
- [ ] Generation produces correct text with world_size > 1
