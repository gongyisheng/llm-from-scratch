# LLM From Scratch

From-scratch PyTorch implementations of popular LLM architectures. Each model is a self-contained tutorial — pick one and follow along.

No `transformers` library. Just raw tensor operations so you understand every line.

## Models

| Model | Key Concepts | Status |
|---|---|---|
| [Qwen3](qwen3/) (0.6B, 1.7B, 4B) | GQA, QK-Norm, RoPE, SwiGLU, KV Cache | ✅ |
| Qwen3-MoE | Mixture of Experts, Shared Experts, Router | 🔜 |
| DeepSeek-V3 | Multi-head Latent Attention, MoE | 🔜 |

## How to Use

All commands run from the project root:

```bash
# 1. Download a model checkpoint
bash scripts/download_qwen3.sh              # default: Qwen3-0.6B
bash scripts/download_qwen3.sh Qwen3-4B     # or pick a larger model

# 2. Run inference
uv run python -m qwen3.main                                              # default: Qwen3-0.6B
uv run python -m qwen3.main -m Qwen3-4B -p "Explain quantum computing"  # pick model + prompt
```

## Architecture Comparison

| Component | Qwen3 | Qwen3-MoE | DeepSeek-V3 |
|---|---|---|---|
| Attention | GQA + QK-Norm | GQA + QK-Norm | MLA |
| Position Encoding | RoPE | RoPE | RoPE (YaRN) |
| FFN | SwiGLU | MoE + Shared Expert | MoE (DeepSeekMoE) |
| Normalization | RMSNorm | RMSNorm | RMSNorm |

## Project Structure

```
llm-from-scratch/
├── pyproject.toml              # shared dependencies
├── README.md
├── scripts/                    # download scripts
│   └── download_qwen3.sh
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
│       └── test_generate.py
└── ...                         # more models follow same structure
```

## Tests

```bash
uv run python -m pytest tests/ -v -m slow
```
