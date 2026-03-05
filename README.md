# LLM From Scratch

From-scratch PyTorch implementations of popular LLM architectures. Each model is a self-contained tutorial — pick one and follow along.

No `transformers` library. Just raw tensor operations so you understand every line.

## Models

| Model | Key Concepts | Status |
|---|---|---|
| [Qwen3](qwen3/) (0.6B–32B) | GQA, QK-Norm, RoPE, SwiGLU, KV Cache | ✅ |
| [Qwen3-MoE](qwen3_moe/) (30B-A3B) | Mixture of Experts, Router | ✅ |
| [GPT-OSS](gpt_oss/) (20B, 120B) | Attention Sinks, YaRN RoPE, Sliding Window, Clamped SwiGLU, MXFP4 | ✅ |
| Qwen3 (next) | | 🔜 |
| Qwen3.5 | | 🔜 |
| Qwen3.5-MoE | | 🔜 |
| DeepSeek-V3 | Multi-head Latent Attention, MoE | 🔜 |

## How to Use

All commands run from the project root:

```bash
# 1. Download a model checkpoint
bash scripts/download.sh Qwen/Qwen3-0.6B          # Qwen3 dense (0.6B–32B)
bash scripts/download.sh Qwen/Qwen3-30B-A3B       # Qwen3-MoE
bash scripts/download.sh openai/gpt-oss-20b        # GPT-OSS

# 2. Run inference
uv run python -m qwen3.main                                                        # Qwen3-0.6B
uv run python -m qwen3.main -m Qwen3-4B -p "Explain quantum computing" --thinking  # with thinking
uv run python -m qwen3_moe.main                                                    # Qwen3-MoE 30B-A3B
uv run python -m gpt_oss.main -p "Tell me about the moon" -n 4096                  # GPT-OSS 20B
```

Common flags: `-m MODEL`, `-p PROMPT`, `-t TEMPERATURE`, `-k TOP_K`, `-n MAX_TOKENS`, `-d DEVICE`.
`--thinking` enables thinking mode (Qwen3 / Qwen3-MoE only).

## What Each Model Adds

All three share GQA, RoPE, SwiGLU, RMSNorm, and KV cache. What's unique:

- **Qwen3** — QK-Norm (RMSNorm per head on Q and K)
- **Qwen3-MoE** — Sparse MoE routing (128 experts, top-8) replacing the dense FFN; fused 3D expert weight tensors, softmax-before-topk routing (matches HF transformers 5.x)
- **GPT-OSS** — Attention sinks, alternating sliding/full window, YaRN RoPE (4K → 131K), clamped SwiGLU, MXFP4 weight quantization, `tiktoken` tokenizer

## Project Structure

Each model is a self-contained package with the same file layout. Shared utilities live at the top level.

```
├── qwen3/                      # Qwen3 dense (0.6B–32B)
│   ├── config.py               # model config dataclass
│   ├── tokenizer.py            # tokenizer + chat template
│   ├── layers.py               # attention, FFN, RMSNorm, RoPE
│   ├── model.py                # full transformer
│   ├── weights.py              # HF safetensors → state dict
│   ├── generate.py             # KV-cache inference loop
│   ├── main.py                 # CLI entry point
│   └── README.md
├── qwen3_moe/                  # Qwen3-MoE (30B-A3B, 235B-A22B)
│   └── (same layout)
├── gpt_oss/                    # GPT-OSS (20B, 120B)
│   └── (same layout)
├── scripts/
│   ├── download.sh             # download model checkpoints
│   ├── compare_hf.py           # token-level accuracy diff vs HF
│   └── inspect_model.py        # inspect checkpoint structure
├── tests/
│   ├── knowledge_runner.py     # shared: knowledge + batch tests
│   ├── accuracy_runner.py      # shared: token-level vs HF tests
│   ├── qwen3/
│   ├── qwen3_moe/
│   └── gpt_oss/
└── checkpoints/                # model weights (gitignored)
```

## Tests

```bash
uv run python -m pytest tests/ -v -m slow
```

## Dependencies

Managed by [uv](https://docs.astral.sh/uv/). Runtime needs only `torch`, `safetensors`, and `tokenizers`. Test extras add `transformers>=5.2`, `accelerate`, `huggingface_hub`, `tqdm`, and `pytest`.
