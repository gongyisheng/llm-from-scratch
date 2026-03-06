import torch
import torch.nn as nn
import torch.distributed as dist
from parallel.comm import get_world_size
from parallel.tensor import ColumnParallelLinear, ParallelEmbedding
from qwen3.layers import RoPE, RMSNorm, TransformerBlock


class Qwen3Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tok_emb = ParallelEmbedding(config.vocab_size, config.emb_dim)
        self.rope = RoPE(config.head_dim, config.rope_base, config.context_length)
        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.final_norm = RMSNorm(config.emb_dim)
        self.lm_head = ColumnParallelLinear(config.emb_dim, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.tok_emb.weight

    def forward(self, token_ids, position_ids=None, kv_cache=None, attn_mask=None):
        if position_ids is None:
            seq_len = token_ids.shape[1]
            position_ids = torch.arange(seq_len, device=token_ids.device).unsqueeze(0)

        x = self.tok_emb(token_ids)  # (batch, seq, emb_dim)

        new_kv_caches = []
        for i, layer in enumerate(self.layers):
            layer_kv_cache = kv_cache[i] if kv_cache else None
            x, new_layer_kv_cache = layer(
                x, self.rope, position_ids, layer_kv_cache, attn_mask
            )
            new_kv_caches.append(new_layer_kv_cache)

        x = self.final_norm(x)
        logits = self.lm_head(x)  # (batch, seq, vocab_size) or (batch, seq, vocab_size // world_size)

        # gather split logits from all ranks
        world_size = get_world_size()
        if world_size > 1:
            all_logits = [torch.zeros_like(logits) for _ in range(world_size)]
            dist.all_gather(all_logits, logits)
            logits = torch.cat(all_logits, dim=-1)  # (batch, seq, vocab_size)

        return logits, new_kv_caches
