from pathlib import Path

import torch
from safetensors.torch import load_file

# FP4 E2M1 lookup table: nibble (0-15) -> float value
# Layout: [sign, exp1, exp0, mantissa], bias=1
FP4_TABLE = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,   # positive
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,  # negative
])

# HF key -> model state_dict key (global)
KEY_MAP = {
    "model.embed_tokens.weight": "tok_emb.weight",
    "model.norm.weight": "final_norm.weight",
    "lm_head.weight": "lm_head.weight",
}

# HF per-layer suffix -> model per-layer suffix (bf16 keys)
LAYER_KEY_MAP = {
    "input_layernorm.weight": "norm1.weight",
    "post_attention_layernorm.weight": "norm2.weight",
    "self_attn.o_proj.weight": "attn.out.weight",
    "self_attn.o_proj.bias": "attn.out.bias",
    "mlp.router.weight": "moe_ffn.router.proj.weight",
    "mlp.router.bias": "moe_ffn.router.proj.bias",
    "mlp.experts.gate_up_proj_bias": "moe_ffn.gate_up_proj_bias",
    "mlp.experts.down_proj_bias": "moe_ffn.down_proj_bias",
}

# Separate Q/K/V projections in checkpoint -> fused qkv_proj in model
QKV_SUFFIXES = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]

# MXFP4 quantized pairs: (blocks_suffix, scales_suffix) -> model weight key
MXFP4_PAIRS = {
    "mlp.experts.gate_up_proj": "moe_ffn.gate_up_proj_weight",
    "mlp.experts.down_proj": "moe_ffn.down_proj_weight",
}


def decompress_mxfp4(blocks: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Decompress MXFP4 (E2M1 values + E8M0 scales) to bf16.

    Args:
        blocks: U8 tensor (..., n_blocks, 16) — two 4-bit values per byte
        scales: U8 tensor (..., n_blocks) — E8M0 shared exponent per block
    Returns:
        bf16 tensor (..., n_blocks * 32)
    """
    # Extract low and high nibbles from each byte
    low = blocks & 0x0F               # (..., n_blocks, 16)
    high = (blocks >> 4) & 0x0F       # (..., n_blocks, 16)

    # Interleave: [low0, high0, low1, high1, ...] -> (..., n_blocks, 32)
    nibbles = torch.stack([low, high], dim=-1).reshape(*blocks.shape[:-1], 32)

    # Lookup FP4 values
    lut = FP4_TABLE.to(device=blocks.device, dtype=torch.bfloat16)
    values = lut[nibbles.long()]      # (..., n_blocks, 32)

    # E8M0 scale: 2^(scale_byte - 127)
    scale_values = torch.pow(
        2.0, scales.to(torch.float32) - 127.0
    ).to(torch.bfloat16)
    values = values * scale_values.unsqueeze(-1)  # broadcast across 32 values

    # Flatten blocks: (..., n_blocks, 32) -> (..., n_blocks * 32)
    return values.reshape(*values.shape[:-2], -1)


def rename_hf_key(hf_key: str) -> str | None:
    if hf_key in KEY_MAP:
        return KEY_MAP[hf_key]

    if not hf_key.startswith("model.layers."):
        return None

    parts = hf_key.split(".", 3)  # ["model", "layers", "idx", suffix]
    layer_idx = parts[2]
    hf_suffix = parts[3]

    if hf_suffix in LAYER_KEY_MAP:
        return f"layers.{layer_idx}.{LAYER_KEY_MAP[hf_suffix]}"

    return None


def load_weights(model, model_dir: str | Path, dtype: torch.dtype | None = None):
    model_dir = Path(model_dir)
    loaded = 0

    for f in sorted(model_dir.glob("*.safetensors")):
        shard = load_file(str(f))
        renamed = {}
        mxfp4_cache = {}  # model_key -> {"blocks": ..., "scales": ...}
        qkv_cache = {}    # "layers.{i}" -> {"q_proj.weight": ..., ...}

        for hf_key, tensor in shard.items():
            if not hf_key.startswith("model.layers."):
                # Global keys
                new_key = rename_hf_key(hf_key)
                if new_key is None:
                    print(f"skipping unknown key: {hf_key}")
                    continue
                if dtype is not None:
                    tensor = tensor.to(dtype=dtype)
                renamed[new_key] = tensor
                loaded += 1
                continue

            parts = hf_key.split(".", 3)  # ["model", "layers", "idx", suffix]
            layer_idx = parts[2]
            hf_suffix = parts[3]

            # MXFP4 quantized expert weights
            matched_mxfp4 = False
            for prefix, model_suffix in MXFP4_PAIRS.items():
                if hf_suffix == f"{prefix}_blocks":
                    cache_key = f"layers.{layer_idx}.{model_suffix}"
                    mxfp4_cache.setdefault(cache_key, {})["blocks"] = tensor
                    matched_mxfp4 = True
                    break
                elif hf_suffix == f"{prefix}_scales":
                    cache_key = f"layers.{layer_idx}.{model_suffix}"
                    mxfp4_cache.setdefault(cache_key, {})["scales"] = tensor
                    matched_mxfp4 = True
                    break
            if matched_mxfp4:
                continue

            # Separate Q/K/V projections -> collect for fusing
            matched_qkv = False
            for qkv_prefix in QKV_SUFFIXES:
                if hf_suffix.startswith(qkv_prefix):
                    layer_key = f"layers.{layer_idx}"
                    # e.g. "self_attn.q_proj.weight" -> "q_proj.weight"
                    part_name = hf_suffix.replace("self_attn.", "")
                    qkv_cache.setdefault(layer_key, {})[part_name] = tensor
                    matched_qkv = True
                    break
            if matched_qkv:
                continue

            # Sinks: reshape (n_heads,) -> (1, n_heads, 1, 1)
            if hf_suffix == "self_attn.sinks":
                renamed[f"layers.{layer_idx}.attn.sinks"] = tensor.reshape(1, -1, 1, 1)
                loaded += 1
                continue

            # Normal layer keys
            if hf_suffix in LAYER_KEY_MAP:
                new_key = f"layers.{layer_idx}.{LAYER_KEY_MAP[hf_suffix]}"
                if dtype is not None:
                    tensor = tensor.to(dtype=dtype)
                renamed[new_key] = tensor
                loaded += 1
            else:
                print(f"skipping unknown key: {hf_key}")

        # Fuse Q/K/V into qkv_proj
        for layer_key, qkv_parts in qkv_cache.items():
            for param_type in ["weight", "bias"]:
                parts = []
                for prefix in ["q_proj", "k_proj", "v_proj"]:
                    key = f"{prefix}.{param_type}"
                    if key in qkv_parts:
                        parts.append(qkv_parts[key])
                if parts:
                    fused = torch.cat(parts, dim=0)
                    if dtype is not None:
                        fused = fused.to(dtype=dtype)
                    renamed[f"{layer_key}.attn.qkv_proj.{param_type}"] = fused
                    loaded += 1

        # Decompress MXFP4 pairs
        for model_key, pair in mxfp4_cache.items():
            if "blocks" in pair and "scales" in pair:
                weight = decompress_mxfp4(pair["blocks"], pair["scales"])
                if dtype is not None:
                    weight = weight.to(dtype=dtype)
                renamed[model_key] = weight
                loaded += 1

        model.load_state_dict(renamed, strict=False, assign=True)
        del shard, renamed, mxfp4_cache, qkv_cache

    print(f"loaded {loaded} weights")


if __name__ == "__main__":
    from gpt_oss.config import GPTOSSConfig
    from gpt_oss.model import GPTOSSModel

    _root = Path(__file__).resolve().parent.parent
    config = GPTOSSConfig.from_model_dir(_root / "checkpoints/gpt-oss-20b")
    with torch.device("meta"):
        model = GPTOSSModel(config)
    load_weights(model, _root / "checkpoints/gpt-oss-20b", dtype=config.dtype)
