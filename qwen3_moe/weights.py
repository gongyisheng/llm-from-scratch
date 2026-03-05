import re
from pathlib import Path

import torch
import torch.nn as nn
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
    "mlp.gate.weight": "moe_ffn.router.proj.weight",
}

# Pattern for per-expert weight keys in safetensors checkpoints
_EXPERT_RE = re.compile(
    r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight"
)


def rename_hf_key(hf_key: str) -> str | None:
    if hf_key in KEY_MAP:
        return KEY_MAP[hf_key]

    if not hf_key.startswith("model.layers."):
        return None

    parts = hf_key.split(".", 3)  # ["model", "layers", "0", suffix]
    layer_idx = parts[2]
    hf_suffix = parts[3]

    # simple layer mappings (attention, norms, router)
    if hf_suffix in LAYER_KEY_MAP:
        return f"layers.{layer_idx}.{LAYER_KEY_MAP[hf_suffix]}"

    # expert keys are fused into 3D tensors in load_weights
    return None


def load_weights(model, model_dir: str | Path, dtype: torch.dtype | None = None):
    model_dir = Path(model_dir)
    loaded = 0
    expert_parts = {}  # {layer_idx: {(expert_idx, proj_type): tensor}}

    for f in sorted(model_dir.glob("*.safetensors")):
        shard = load_file(str(f))
        renamed = {}
        for hf_key, tensor in shard.items():
            if dtype is not None:
                tensor = tensor.to(dtype=dtype)

            # expert weights: buffer for 3D fusion
            m = _EXPERT_RE.match(hf_key)
            if m:
                layer_idx = int(m.group(1))
                expert_idx = int(m.group(2))
                proj_type = m.group(3)  # gate_proj, up_proj, or down_proj
                expert_parts.setdefault(layer_idx, {})[(expert_idx, proj_type)] = tensor
                loaded += 1
                continue

            new_key = rename_hf_key(hf_key)
            if new_key is None:
                print(f"skipping unknown key: {hf_key}")
                continue
            renamed[new_key] = tensor
            loaded += 1
        model.load_state_dict(renamed, strict=False, assign=True)
        del shard, renamed

    # fuse per-expert weights into 3D tensors (gate_up_proj, down_proj)
    for layer_idx in sorted(expert_parts):
        parts = expert_parts[layer_idx]
        block = model.layers[layer_idx].moe_ffn
        gate_up = torch.stack([
            torch.cat([parts[(j, "gate_proj")], parts[(j, "up_proj")]], dim=0)
            for j in range(block.n_experts)
        ])  # (n_experts, 2 * moe_hidden_dim, emb_dim)
        down = torch.stack([
            parts[(j, "down_proj")] for j in range(block.n_experts)
        ])  # (n_experts, emb_dim, moe_hidden_dim)
        block.gate_up_proj = nn.Parameter(gate_up)
        block.down_proj = nn.Parameter(down)
    del expert_parts

    # assign=True replaces Parameter objects, breaking weight tying.
    # Re-tie lm_head to tok_emb when the checkpoint omits lm_head.weight.
    if hasattr(model, "lm_head") and hasattr(model, "tok_emb"):
        if model.lm_head.weight.is_meta and not model.tok_emb.weight.is_meta:
            model.lm_head.weight = model.tok_emb.weight


if __name__ == "__main__":
    from qwen3_moe.config import Qwen3MoEConfig
    from qwen3_moe.model import Qwen3MoEModel

    _root = Path(__file__).resolve().parent.parent
    config = Qwen3MoEConfig.from_model_dir(_root / "checkpoints/Qwen3-30B-A3B")
    with torch.device("meta"):
        model = Qwen3MoEModel(config)
    load_weights(model, _root / "checkpoints/Qwen3-30B-A3B", dtype=config.dtype)
