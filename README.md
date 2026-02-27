# LLM From Scratch

From-scratch PyTorch implementations of popular LLM architectures. Each model is a self-contained tutorial — pick one and follow along.

No `transformers` library. Just raw tensor operations so you understand every line.

## Models

| Model | Key Concepts | Status |
|---|---|---|
| [Qwen3](qwen3/) (0.6B, 1.7B, 4B) | GQA, QK-Norm, RoPE, SwiGLU, KV Cache | ✅ |
| [Qwen3-MoE](qwen3_moe/) (30B-A3B) | Mixture of Experts, Router | ✅ |
| [GPT-OSS](gpt_oss/) (20B) | Attention Sinks, YaRN RoPE, Sliding Window, Clamped SwiGLU, MXFP4 | 🔧 |
| DeepSeek-V3 | Multi-head Latent Attention, MoE | 🔜 |

## How to Use

All commands run from the project root:

```bash
# 1. Download a model checkpoint
bash scripts/download.sh              # default: Qwen/Qwen3-0.6B
bash scripts/download.sh Qwen/Qwen3-4B

# 2. Run inference
uv run python -m qwen3.main                                              # default: Qwen3-0.6B
uv run python -m qwen3.main -m Qwen3-4B -p "Explain quantum computing"  # pick model + prompt
```

## Architecture Comparison

| Component | Qwen3 | Qwen3-MoE | GPT-OSS | DeepSeek-V3 |
|---|---|---|---|---|
| Attention | GQA + QK-Norm | GQA + QK-Norm | GQA + Sinks + Sliding Window | MLA |
| Position Encoding | RoPE | RoPE | YaRN RoPE | RoPE (YaRN) |
| FFN | SwiGLU | MoE (128 experts, top-8) | MoE (32 experts, top-4, clamped SwiGLU) | MoE (DeepSeekMoE) |
| Normalization | RMSNorm | RMSNorm | RMSNorm | RMSNorm |

## Project Structure

```
llm-from-scratch/
├── pyproject.toml              # shared dependencies
├── README.md
├── scripts/
│   ├── download.sh             # download model checkpoints
│   └── compare_hf.py           # layer-by-layer accuracy diff: scratch vs HuggingFace eager
├── checkpoints/                # model weights (gitignored)
├── qwen3/                      # Qwen3 implementation
│   ├── README.md
│   ├── config.py
│   ├── tokenizer.py
│   ├── layers.py
│   ├── model.py
│   ├── weights.py
│   ├── generate.py
│   └── main.py
├── tests/                      # tests for all models
│   └── qwen3/
│       └── test_knowledge.py
└── ...                         # more models follow same structure
```

## Tests

```bash
uv run python -m pytest tests/ -v -m slow
```
