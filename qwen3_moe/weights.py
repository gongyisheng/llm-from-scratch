from pathlib import Path
import torch
from safetensors.torch import load_file

# HF key -> model state_dict key
KEY_MAP = {
    "model.embed_tokens.weight": "tok_emb.weight",
    "model.norm.weight": "final_norm.weight",
    "lm_head.weight": "lm_head.weight",
}

# per-layer: HF suffix -> model suffix
LAYER_KEY_MAP = {
    "input_layernorm.weight": "norm1.weight",
    "post_attention_layernorm.weight": "norm2.weight",
    "self_attn.q_proj.weight": "attn.q_proj.weight",
    "self_attn.k_proj.weight": "attn.k_proj.weight",
    "self_attn.v_proj.weight": "attn.v_proj.weight",
    "self_attn.o_proj.weight": "attn.o_proj.weight",
    "self_attn.q_norm.weight": "attn.q_norm.weight",
    "self_attn.k_norm.weight": "attn.k_norm.weight",
    "mlp.gate.weight": "moe_ffn.gate.proj.weight",
}


def rename_hf_key(hf_key: str) -> str | None:
    if hf_key in KEY_MAP:
        return KEY_MAP[hf_key]

    if not hf_key.startswith("model.layers."):
        return None

    parts = hf_key.split(".", 3)  # ["model", "layers", "0", suffix]
    layer_idx = parts[2]
    hf_suffix = parts[3]

    # simple layer mappings (attention, norms, gate)
    if hf_suffix in LAYER_KEY_MAP:
        return f"layers.{layer_idx}.{LAYER_KEY_MAP[hf_suffix]}"

    # MoE experts: mlp.experts.{j}.gate_proj.weight -> moe_ffn.experts.{j}.gate_proj.weight
    if hf_suffix.startswith("mlp.experts."):
        model_suffix = hf_suffix.replace("mlp.experts.", "moe_ffn.experts.", 1)
        return f"layers.{layer_idx}.{model_suffix}"

    return None


def load_weights(model, model_dir: str | Path, dtype: torch.dtype | None = None):
    model_dir = Path(model_dir)
    loaded = 0
    for f in sorted(model_dir.glob("*.safetensors")):
        shard = load_file(str(f))
        renamed = {}
        for hf_key, tensor in shard.items():
            new_key = rename_hf_key(hf_key)
            if new_key is None:
                print(f"skipping unknown key: {hf_key}")
                continue
            if dtype is not None:
                tensor = tensor.to(dtype=dtype)
            renamed[new_key] = tensor
            loaded += 1
        model.load_state_dict(renamed, strict=False, assign=True)
        del shard, renamed
    print(f"loaded {loaded} weights")

if __name__ == "__main__":
    from qwen3_moe.config import Qwen3MoEConfig
    from qwen3_moe.model import Qwen3MoEModel

    _root = Path(__file__).resolve().parent.parent
    config = Qwen3MoEConfig.from_model_dir(_root / "checkpoints/Qwen3-30B-A3B")
    with torch.device("meta"):
        model = Qwen3MoEModel(config)
    load_weights(model, _root / "checkpoints/Qwen3-30B-A3B", dtype=config.dtype)

