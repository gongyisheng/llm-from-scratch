import json
from dataclasses import dataclass
from pathlib import Path

import torch

DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}

@dataclass
class GPTOSSConfig:
    vocab_size: int
    emb_dim: int  # hidden_size
    n_heads: int  # num_attn_heads
    n_kv_groups: int  # num_kv_heads
    head_dim: int
    n_layers: int  # num_hidden_layers
    n_experts: int # num_experts
    n_experts_per_token: int # num_experts_per_tok
    moe_hidden_dim: int # intermediate_size
    moe_norm_topk_prob: bool # norm_topk_prob
    swiglu_limit: float
    context_length: int  # max_position_embedding
    context_sliding_window: int # sliding_window
    rope_base: float  # rope_theta
    yarn_original_context_length: int
    yarn_scaling_factor: float
    yarn_beta_slow: float
    yarn_beta_fast: float
    rms_norm_eps: float
    tie_word_embeddings: bool
    dtype: torch.dtype
    eos_token_id: int
    group_size: int

    @classmethod
    def from_model_dir(cls, path: str | Path) -> "GPTOSSConfig":
        with open(Path(path) / "config.json") as f:
            cfg = json.load(f)
        n_heads = cfg["num_attention_heads"]
        n_kv_groups = cfg["num_key_value_heads"]
        rope_scaling = cfg.get("rope_scaling", {})
        return cls(
            vocab_size=cfg["vocab_size"],
            emb_dim=cfg["hidden_size"],
            n_heads=n_heads,
            n_kv_groups=n_kv_groups,
            head_dim=cfg["head_dim"],
            n_layers=cfg["num_hidden_layers"],
            n_experts=cfg["num_local_experts"],
            n_experts_per_token=cfg["num_experts_per_tok"],
            moe_hidden_dim=cfg["intermediate_size"],
            moe_norm_topk_prob=True,
            swiglu_limit=cfg["swiglu_limit"],
            context_length=cfg["max_position_embeddings"],
            context_sliding_window=cfg["sliding_window"],
            rope_base=cfg["rope_theta"],
            yarn_original_context_length=cfg["initial_context_length"],
            yarn_scaling_factor=rope_scaling.get("factor", 1.0),
            yarn_beta_slow=rope_scaling.get("beta_slow", 1.0),
            yarn_beta_fast=rope_scaling.get("beta_fast", 32.0),
            rms_norm_eps=cfg["rms_norm_eps"],
            tie_word_embeddings=cfg["tie_word_embeddings"],
            dtype=DTYPE_MAP.get(cfg.get("torch_dtype", "bfloat16"), torch.bfloat16),
            eos_token_id=cfg["eos_token_id"],
            group_size=n_heads // n_kv_groups,
        )

if __name__ == "__main__":
    _root = Path(__file__).resolve().parent.parent
    config = GPTOSSConfig.from_model_dir(_root / "checkpoints/gpt-oss-20b")
    print(config)
