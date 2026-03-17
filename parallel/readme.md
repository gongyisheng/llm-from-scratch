# Tensor Parallelism

Step-by-step guide covering what was implemented for **Qwen3** and what's planned for **Qwen3-MoE** and **GPT-OSS**.

## Background

### Core Principle

Tensor parallelism always pairs **Column → Row**:
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

## Qwen3 (Done)

### File Structure

```
parallel/
├── comm.py        # init_process_group, get_rank, get_world_size, all_reduce, destroy_process_group
├── tensor.py      # ColumnParallelLinear, RowParallelLinear, ParallelEmbedding, shard_tensor
└── __init__.py    # empty

qwen3/
├── layers.py      # SwiGLUFFN, GroupQueryAttention natively use Column/RowParallelLinear
├── model.py       # Qwen3Model uses ParallelEmbedding + all_gather on logits
├── weights.py     # load_weights shards tensors per rank via shard_tensor
├── generate.py    # broadcast after sample() for parallel sync
└── main.py        # torchrun detection, unified single/parallel flow
```

### Step 1: Communication Primitives (`parallel/comm.py`)

```python
def init_process_group(backend: str = "auto"):
    # reads RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT from env (set by torchrun)
    # backend="auto": nccl if device_count >= world_size, else gloo
    # assigns each rank to its own GPU via LOCAL_RANK when enough GPUs exist
```

Key functions: `init_process_group`, `get_rank`, `get_world_size`, `all_reduce`, `destroy_process_group`.

Backend auto-detection:
- **nccl** when `torch.cuda.device_count() >= world_size` (one GPU per rank)
- **gloo** otherwise (single GPU or CPU-only; nccl requires one GPU per rank)

### Step 2: Parallel Tensor Primitives (`parallel/tensor.py`)

**ColumnParallelLinear** — splits output dim across ranks, no communication:
```python
class ColumnParallelLinear(nn.Module):
    # weight shape: (out_features // world_size, in_features)
    # gets world_size from get_world_size() at init time
    def forward(self, x):
        return F.linear(x, self.weight, self.bias)
```

**RowParallelLinear** — splits input dim across ranks, all_reduce in forward:
```python
class RowParallelLinear(nn.Module):
    # weight shape: (out_features, in_features // world_size)
    def forward(self, x):
        return all_reduce(F.linear(x, self.weight, self.bias))
```

**ParallelEmbedding** — splits vocab rows across ranks:
```python
class ParallelEmbedding(nn.Module):
    # each rank stores vocab_size // world_size rows
    # tokens outside this rank's range produce zeros; all_reduce combines
    def forward(self, x):
        mask = (x >= self.start) & (x < self.end)
        local_ids = (x - self.start).clamp(0, self.embd_per_rank - 1)
        output = F.embedding(local_ids, self.weight) * mask.unsqueeze(-1)
        return all_reduce(output)
```

**shard_tensor** — slice a tensor along any dim for the current rank:
```python
def shard_tensor(tensor, dim):
    # uses get_rank() and get_world_size() internally
    return tensor.narrow(dim, start, shard_size).contiguous()
```

### Step 3: Parallel Layers (`qwen3/layers.py`)

Added at the bottom of the existing `layers.py` file (not a separate file).

**ParallelGroupQueryAttention**:
```
x (replicated)
  |
  q_proj: ColumnParallel  -- split heads across ranks
  k_proj: ColumnParallel  -- split kv groups across ranks
  v_proj: ColumnParallel  -- split kv groups across ranks
  |
  [q_norm, k_norm, rope, kv_cache, attention — each rank works on its local heads]
  |
  o_proj: RowParallel      -- all_reduce to recombine
  |
x (replicated)
```

- `n_heads` and `n_kv_groups` are divided by `world_size` at init
- `q_norm`, `k_norm` are unchanged (per-head, head_dim is the same)
- Constraint: `n_heads % world_size == 0` and `n_kv_groups % world_size == 0`

**ParallelSwiGLUFFN**:
```
x (replicated)
  |
  gate_proj: ColumnParallel  -- split hidden across ranks
  up_proj:   ColumnParallel  -- split hidden across ranks
  |
  silu(gate) * up  -- element-wise on split hidden
  |
  down_proj: RowParallel     -- all_reduce to recombine
  |
x (replicated)
```

**ParallelTransformersBlock**: wraps `ParallelGroupQueryAttention` + `ParallelSwiGLUFFN` with pre-norm + residual. Same structure as the non-parallel `TransformerBlock`.

### Step 4: Parallel Model (`qwen3/model.py`)

`ParallelQwen3Model` added at the bottom of existing `model.py`:

```python
class ParallelQwen3Model(nn.Module):
    def __init__(self, config):
        self.tok_emb = ParallelEmbedding(config.vocab_size, config.emb_dim)
        self.rope = RoPE(...)           # replicated
        self.layers = [ParallelTransformersBlock(config) for ...]
        self.final_norm = RMSNorm(...)  # replicated
        self.lm_head = ColumnParallelLinear(config.emb_dim, config.vocab_size, bias=False)
        # weight tying works the same way

    def forward(self, token_ids, position_ids, kv_cache=None, attn_mask=None):
        # ... same flow as Qwen3Model ...
        logits = self.lm_head(x)  # (batch, seq, vocab_size // world_size)

        # gather split logits so caller gets full vocab
        if world_size > 1:
            all_logits = [torch.zeros_like(logits) for _ in range(world_size)]
            dist.all_gather(all_logits, logits)
            logits = torch.cat(all_logits, dim=-1)  # (batch, seq, vocab_size)

        return logits, new_kv_cache
```

Key decision: **all_gather logits inside the model** so `generate()` works without changes.

### Step 5: Parallel Weight Loading (`qwen3/weights.py`)

`load_parallel_weights` added at the bottom of existing `weights.py`:

```python
COLUMN_PARALLEL_SUFFIXES = (
    "q_proj.weight", "k_proj.weight", "v_proj.weight",
    "gate_proj.weight", "up_proj.weight",
    "tok_emb.weight", "lm_head.weight",
)
ROW_PARALLEL_SUFFIXES = ("o_proj.weight", "down_proj.weight")

def load_parallel_weights(model, model_dir, dtype=None):
    for f in sorted(model_dir.glob("*.safetensors")):
        shard = load_file(str(f))
        for hf_key, tensor in shard.items():
            new_key = rename_hf_key(hf_key)  # reuses existing key mapping
            if new_key.endswith(COLUMN_PARALLEL_SUFFIXES):
                tensor = shard_tensor(tensor, 0)   # shard output dim
            elif new_key.endswith(ROW_PARALLEL_SUFFIXES):
                tensor = shard_tensor(tensor, 1)   # shard input dim
            # norms, q_norm, k_norm: replicated (no sharding)
            ...
        model.load_state_dict(renamed, strict=False, assign=True)
    # re-tie lm_head if checkpoint omits it
```

Weight sharding rules:

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

General rule:
- **ColumnParallel**: shard dim=0
- **RowParallel**: shard dim=1
- **ParallelEmbedding**: shard dim=0
- **Everything else**: replicated

### Step 6: Generation Sync (`qwen3/generate.py`)

Sampling is stochastic (`torch.multinomial`), so each rank would sample a different token. Fix: **broadcast from rank 0** after every `sample()` call.

```python
next_token = sample(logits[:, -1, :], ...)
if dist.is_initialized():
    if next_token.dim() == 0:
        next_token = next_token.unsqueeze(0)
    dist.broadcast(next_token, src=0)
```

This is done in both the prefill and decode loop.

### Step 7: Unified Main Flow (`qwen3/main.py`)

`main()` uses `dist.is_initialized()` to pick the right model loader:

```python
def main():
    from parallel.comm import init_process_group, destroy_process_group, get_rank, get_world_size

    # torchrun sets RANK env var — init distributed if present
    if "RANK" in os.environ:
        init_process_group()

    if dist.is_initialized():
        model, tokenizer, config = load_parallel_model(model_dir)
    else:
        model, tokenizer, config = load_model(model_dir, device=args.device)

    # shared inference path — generate() works for both
    output_text = run_inference(model, tokenizer, config, ...)

    if dist.is_initialized():
        destroy_process_group()
```

`load_parallel_model` moves the model to the assigned GPU only when enough GPUs exist:
```python
if torch.cuda.is_available() and torch.cuda.device_count() >= get_world_size():
    model = model.to(torch.cuda.current_device())
```

### Step 8: Tests

Tests reuse existing `test_knowledge.py` and `test_accuracy.py` — no separate parallel test files.

**How it works**: `tests/conftest.py` calls `init_process_group()` when `RANK` is in env. Then test files swap the loader at import time:

```python
if dist.is_initialized():
    from qwen3.main import load_parallel_model as load_model
else:
    from qwen3.main import load_model
```

**Running parallel tests**:
```bash
PYTHONPATH=. uv run torchrun --nproc_per_node=2 -m pytest tests/qwen3/test_knowledge.py \
    -m slow -v -s --model Qwen3-0.6B --device cpu
PYTHONPATH=. uv run torchrun --nproc_per_node=2 -m pytest tests/qwen3/test_accuracy.py \
    -m slow -v -s --model Qwen3-0.6B --device cpu
```

Accuracy tests use `mismatch_expected=True` for parallel mode because tensor parallelism changes computation order (split matmul + all_reduce), causing float differences in bf16.

### Lessons Learned

1. **Connection closed by peer**: `sample()` with temperature > 0 uses `torch.multinomial` (stochastic), so each rank sampled different tokens. One rank hit EOS before the other, destroyed its process group, and the other got "connection closed". Fix: `dist.broadcast(next_token, src=0)` after each sample.

2. **NCCL duplicate GPU error**: NCCL does not allow multiple ranks on the same GPU. Running `torchrun --nproc_per_node=2` on a single-GPU machine fails. Fix: auto-detect backend — fall back to gloo when `device_count < world_size`.

3. **Accuracy test deadlock**: Rank 1 reached `all_reduce` (inside parallel model) while rank 0 was still doing HuggingFace reference inference. Fix: `dist.barrier()` before the parallel inference in accuracy tests.

4. **bf16 accuracy mismatches**: Tensor parallelism splits matmuls differently (partial sums + all_reduce vs full matmul), producing different float rounding in bf16. These are expected and the accuracy test accommodates them with `mismatch_expected=True`.

5. **dist.is_initialized() at import time**: Test files check `dist.is_initialized()` to choose the model loader. This works because `conftest.py` runs before test modules are imported, calling `init_process_group()` when under torchrun.

---

## Qwen3-MoE

Same attention parallelism as Qwen3. Steps 1-2 (comm.py, tensor.py) are shared infrastructure — no changes needed. The difference is the MoE FFN block replacing dense SwiGLUFFN.

### File Structure

```
qwen3_moe/
├── layers.py      # GQA (same parallel pattern), MoERouter (replicated), SparseMoEBlock (sharded experts)
├── model.py       # Qwen3MoEModel with ParallelEmbedding + all_gather on logits
├── weights.py     # Expert fusion + sharding (fuse first, then shard_tensor)
├── generate.py    # broadcast after sample() for parallel sync
└── main.py        # torchrun detection, unified single/parallel flow
```

### Step 3: Parallel Layers (`qwen3_moe/layers.py`)

**Attention**: identical to Qwen3. ColumnParallel q/k/v, RowParallel o_proj, divide `n_heads` and `n_kv_groups` by `world_size`.

**MoERouter**: stays as regular `nn.Linear(emb_dim, n_experts)`. Must be **replicated** — all ranks need identical routing decisions, otherwise tokens get dispatched to wrong experts.

**ParallelSparseMoEBlock** — tensor parallel within each expert:

Qwen3-MoE uses **fused 3D expert weights**:
- `gate_up_proj`: `(n_experts, 2 * moe_hidden_dim, emb_dim)`
- `down_proj`: `(n_experts, emb_dim, moe_hidden_dim)`

**Strategy**: split `moe_hidden_dim` across ranks (keep all experts on all ranks).

```
x (replicated)
  |
  router.proj (replicated)  -- each rank needs identical routing decisions
  |
  for each active expert:
    gate_up_proj[expert_idx]: shape (2 * moe_hidden_dim // world_size, emb_dim)  -- no comm
    chunk -> silu(gate) * up  -- element-wise on split hidden
    down_proj[expert_idx]: shape (emb_dim, moe_hidden_dim // world_size)         -- no comm yet
    weighted by routing_weight, accumulated into output via index_add_
  |
  all_reduce(output)  -- ONCE after entire loop, not per expert
  |
x (replicated)
```

Why this works: each rank computes a **partial sum** of each expert's output (because `down_proj` input dim is split). The partial sums accumulate into the output tensor, and a single `all_reduce` combines them. This gives exactly **2 all_reduces per layer** (1 attn + 1 MoE), same as the dense Qwen3.

Constraint: `moe_hidden_dim % world_size == 0` (768 / 2 = 384, ok).

### Step 4: Parallel Model (`qwen3_moe/model.py`)

Same pattern as Qwen3, with one difference: `tie_word_embeddings=False` in Qwen3-MoE, so `lm_head` has its own weights (no re-tying after load).

```python
class ParallelQwen3MoEModel(nn.Module):
    def __init__(self, config):
        self.tok_emb = ParallelEmbedding(config.vocab_size, config.emb_dim)
        self.layers = [ParallelMoETransformersBlock(config) for ...]
        self.final_norm = RMSNorm(...)  # replicated
        self.lm_head = ColumnParallelLinear(config.emb_dim, config.vocab_size, bias=False)
        # NO weight tying (tie_word_embeddings=False)

    def forward(self, ...):
        # ... same flow ...
        logits = self.lm_head(x)  # (batch, seq, vocab_size // world_size)
        if world_size > 1:
            all_logits = [torch.zeros_like(logits) for _ in range(world_size)]
            dist.all_gather(all_logits, logits)
            logits = torch.cat(all_logits, dim=-1)
        return logits, new_kv_cache
```

### Step 5: Parallel Weight Loading (`qwen3_moe/weights.py`)

Expert weights are loaded per-expert from HuggingFace, fused into 3D tensors, **then** sharded:

```python
# fuse per-expert weights into 3D
gate_up = torch.stack([
    torch.cat([parts[(j, "gate_proj")], parts[(j, "up_proj")]], dim=0)
    for j in range(n_experts)
])  # (n_experts, 2 * moe_hidden_dim, emb_dim)
gate_up = shard_tensor(gate_up, dim=1)  # shard hidden dim

down = torch.stack([
    parts[(j, "down_proj")] for j in range(n_experts)
])  # (n_experts, emb_dim, moe_hidden_dim)
down = shard_tensor(down, dim=2)  # shard hidden dim
```

Full weight sharding rules:

| Layer | Original shape | Shard rule |
|-------|---------------|-----------|
| `tok_emb.weight` | `(vocab, emb_dim)` | shard dim=0 |
| `attn.q_proj.weight` | `(n_heads*head_dim, emb_dim)` | shard dim=0 |
| `attn.k_proj.weight` | `(n_kv*head_dim, emb_dim)` | shard dim=0 |
| `attn.v_proj.weight` | `(n_kv*head_dim, emb_dim)` | shard dim=0 |
| `attn.o_proj.weight` | `(emb_dim, n_heads*head_dim)` | shard dim=1 |
| `moe_ffn.gate_up_proj` | `(n_experts, 2*moe_hidden_dim, emb_dim)` | shard dim=1 |
| `moe_ffn.down_proj` | `(n_experts, emb_dim, moe_hidden_dim)` | shard dim=2 |
| `moe_ffn.router.proj.weight` | `(n_experts, emb_dim)` | NO shard |
| `lm_head.weight` | `(vocab, emb_dim)` | shard dim=0 |
| `*.norm*.weight` | | NO shard |
| `attn.q_norm.weight` | | NO shard |
| `attn.k_norm.weight` | | NO shard |

### Step 6-8: Generation, Main, Tests

Identical to Qwen3 — broadcast after `sample()`, torchrun detection in main, same test structure with `mismatch_expected=True` for parallel mode.

---

## GPT-OSS (Planned)

Same tensor parallel pattern as Qwen3-MoE, with these differences:

### Attention differences

- **bias=True** on q/k/v/out projections: biases must also be sharded for ColumnParallel (dim=0), but NOT sharded for RowParallel (replicated, summed once via all_reduce)
- **No q_norm/k_norm**: nothing extra to handle
- **Attention sinks**: `self.sinks = nn.Parameter(shape (1, n_heads, 1, 1))` — shard along n_heads dim (dim=1)
- **Sliding window**: no change needed (works per-head, same logic)

### MoE FFN differences

- Expert weights have **biases**: `gate_up_proj_bias (n_experts, moe_hidden_dim*2)` and `down_proj_bias (n_experts, emb_dim)`
- `gate_up_proj_bias`: shard dim=1 (follows gate_up_proj_weight sharding)
- `down_proj_bias`: NO shard (output dim, replicated)
- **MXFP4 decompression**: decompress first, THEN shard

### Weight sharding for GPT-OSS

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

## Checklist Per Model

- [x] Qwen3: `world_size=1` produces identical output to the non-parallel model
- [x] Qwen3: `n_heads % world_size == 0` and `n_kv_groups % world_size == 0`
- [x] Qwen3: `hidden_dim % world_size == 0`
- [x] Qwen3: `vocab_size % world_size == 0`
- [x] Qwen3: All-reduce count matches: 2 per layer (attn o_proj + ffn down_proj) + 1 for embedding
- [x] Qwen3: Weight shapes after sharding match model parameter shapes
- [x] Qwen3: Generation produces correct text with world_size=2
- [ ] Qwen3-MoE: `world_size=1` produces identical output to the non-parallel model
- [ ] Qwen3-MoE: `n_heads % world_size == 0` and `n_kv_groups % world_size == 0`
- [ ] Qwen3-MoE: `moe_hidden_dim % world_size == 0`
- [ ] Qwen3-MoE: `vocab_size % world_size == 0`
- [ ] Qwen3-MoE: All-reduce count: 2 per layer (attn o_proj + MoE after expert loop) + 1 for embedding
- [ ] Qwen3-MoE: Router replicated — identical routing decisions on all ranks
- [ ] Qwen3-MoE: Expert fusion happens before sharding (fuse 3D, then shard_tensor)
- [ ] Qwen3-MoE: Weight shapes after sharding match model parameter shapes
- [ ] Qwen3-MoE: Generation produces correct text with world_size=2
- [ ] GPT-OSS: all of the above + bias sharding + MXFP4 decompression before sharding
