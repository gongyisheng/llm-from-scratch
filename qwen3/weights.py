from pathlib import Path
import torch
from safetensors.torch import load_file

# HF key -> your model's state_dict key
KEY_MAP = {
    "model.embed_tokens.weight": "tok_emb.weight",
    "model.norm.weight": "final_norm.weight",
    "lm_head.weight": "lm_head.weight",
}

# per-layer key mapping: HF suffix -> your suffix
LAYER_KEY_MAP = {
    "input_layernorm.weight": "norm1.weight",
    "post_attention_layernorm.weight": "norm2.weight",
    "self_attn.q_proj.weight": "attn.W_q.weight",
    "self_attn.k_proj.weight": "attn.W_k.weight",
    "self_attn.v_proj.weight": "attn.W_v.weight",
    "self_attn.o_proj.weight": "attn.W_o.weight",
    "self_attn.q_norm.weight": "attn.q_norm.weight",
    "self_attn.k_norm.weight": "attn.k_norm.weight",
    "mlp.gate_proj.weight": "ffn.W_gate.weight",
    "mlp.up_proj.weight": "ffn.W_up.weight",
    "mlp.down_proj.weight": "ffn.W_down.weight",
}


def rename_hf_key(hf_key: str) -> str | None:
    if hf_key in KEY_MAP:
        return KEY_MAP[hf_key]

    # model.layers.{i}.{suffix} -> layers.{i}.{suffix}
    if hf_key.startswith("model.layers."):
        parts = hf_key.split(
            ".", 3
        )  # ["model", "layers", "0", "self_attn.q_proj.weight"]
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
    from qwen3.config import Qwen3Config
    from qwen3.model import Qwen3Model

    _root = Path(__file__).resolve().parent.parent
    config = Qwen3Config.from_model_dir(_root / "checkpoints/Qwen3-0.6B")
    with torch.device("meta"):
        model = Qwen3Model(config)
    load_weights(model, _root / "checkpoints/Qwen3-0.6B", dtype=config.dtype)
