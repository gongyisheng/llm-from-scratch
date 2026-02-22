import torch
import torch.nn as nn

from qwen3_moe.config import Qwen3MoEConfig
from qwen3_moe.layers import RoPE, RMSNorm, MoETransformersBlock

class Qwen3MoEModel(nn.Module):
    def __init__(self, config: Qwen3MoEConfig):
        super().__init__()
        self.tok_emb = nn.Embedding(config.vocab_size, config.emb_dim)
        self.rope = RoPE(config.head_dim, config.rope_base, config.context_length)
        self.layers = nn.ModuleList([
            MoETransformersBlock(config.emb_dim, config.n_heads, config.n_kv_groups, config.head_dim, config.n_experts, config.n_experts_per_token, config.moe_hidden_dim, config.moe_norm_topk_prob) for _ in range(config.n_layers)
        ])
        self.final_norm = RMSNorm(config.emb_dim)
        self.lm_head = nn.Linear(config.emb_dim, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.tok_emb.weight
    
    def forward(self, token_ids, position_ids, kv_cache=None, attn_mask=None):
        if position_ids is None:
            seq_len = token_ids.shape[1]
            position_ids = torch.arange(seq_len, device=token_ids.device).unsqueeze(0)

        x = self.tok_emb(token_ids)

        new_kv_caches = []
        for i, layer in enumerate(self.layers):
            layer_kv_cache = kv_cache[i] if kv_cache else None
            x, new_layer_kv_cache = layer(x, self.rope, position_ids, layer_kv_cache, attn_mask)
            new_kv_caches.append(new_layer_kv_cache)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits, new_kv_caches
        