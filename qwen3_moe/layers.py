import torch
import torch.nn as nn
import torch.nn.functional as F

from parallel.comm import get_world_size
from parallel.tensor import ColumnParallelLinear, RowParallelLinear


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.to(torch.float32)
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


class RoPE(nn.Module):
    def __init__(self, head_dim: int, rope_base: int, max_seq_len: int):
        super().__init__()
        self.head_dim = head_dim
        self.rope_base = rope_base
        self.max_seq_len = max_seq_len
        self._build_buffers()

    def _build_buffers(self):
        freqs = 1.0 / self.rope_base ** (torch.arange(0, self.head_dim, 2) / self.head_dim) # (head_dim/2,)
        positions = torch.arange(0, self.max_seq_len) # (max_seq_len,)
        angles = positions[: ,None] * freqs[None, :] # (max_seq_len, head_dim/2)
        angles = torch.concat([angles, angles], dim=-1) # (max_seq_len, head_dim)
        self.register_buffer("sin", torch.sin(angles))
        self.register_buffer("cos", torch.cos(angles))
    
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        # x.shape: (batch, n_heads, seq_len, head_dim)
        # postion_ids.shape: (batch, seq_len)
        cos = self.cos[position_ids][:, None, :, :].to(x.dtype)  # (batch, 1, seq_len, head_dim)
        sin = self.sin[position_ids][:, None, :, :].to(x.dtype)

        x1 = x[..., :self.head_dim // 2]
        x2 = x[..., self.head_dim // 2:]

        rotated = torch.concat([-x2, x1], dim=-1)
        output = x*cos + rotated*sin
        return output


class GroupQueryAttention(nn.Module):
    def __init__(self, emb_dim: int, n_heads: int, n_kv_groups: int, head_dim: int):
        super().__init__()
        self.emb_dim = emb_dim
        self.head_dim = head_dim
        world_size = get_world_size()
        self.n_heads_local = n_heads // world_size
        self.n_kv_groups_local = n_kv_groups // world_size
        self.group_size = n_heads // n_kv_groups

        self.q_proj = ColumnParallelLinear(self.emb_dim, self.head_dim * n_heads, bias=False)
        self.k_proj = ColumnParallelLinear(self.emb_dim, self.head_dim * n_kv_groups, bias=False)
        self.v_proj = ColumnParallelLinear(self.emb_dim, self.head_dim * n_kv_groups, bias=False)
        self.o_proj = RowParallelLinear(self.head_dim * n_heads, self.emb_dim, bias=False)

        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
    
    def forward(self, x, rope, position_ids, kv_cache=None, attn_mask=None) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # x.shape: (batch, seq, emb_dim)
        batch, seq_len, _ = x.shape

        # projection + reshape to [batch, heads, seq, head_dim]
        # n_heads and n_kv_groups are local counts (already divided by world_size)
        Q = self.q_proj(x).view(batch, seq_len, self.n_heads_local, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch, seq_len, self.n_kv_groups_local, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch, seq_len, self.n_kv_groups_local, self.head_dim).transpose(1, 2)

        # Q/K norm: attention logits can explode, need norm
        Q = self.q_norm(Q)
        K = self.k_norm(K)

        Q = rope(Q, position_ids)
        K = rope(K, position_ids)

        if kv_cache is not None:
            past_k, past_v = kv_cache
            # decode, each step seq_len = 1, concat at seq_len dim
            K = torch.concat([past_k, K], dim=2)
            V = torch.concat([past_v, V], dim=2)
        new_kv_cache = (K, V)

        K = K.repeat_interleave(self.group_size, dim=1) # repeat at n_kv_groups dim
        V = V.repeat_interleave(self.group_size, dim=1) # repeat at n_kv_groups dim

        scores = Q @ K.transpose(-2, -1) / (self.head_dim ** 0.5) # (batch, n_heads, seq_len, total_seq_len)
        if attn_mask is not None:
            scores = scores + attn_mask
        attn = F.softmax(scores, dim=-1, dtype=torch.float32).to(x.dtype)
        context = attn @ V

        context = context.transpose(1,2).reshape(batch, seq_len, -1)
        output = self.o_proj(context)
        return output, new_kv_cache


class MoERouter(nn.Module):
    def __init__(self, emb_dim: int, n_experts: int, n_experts_per_token: int, moe_norm_topk_prob: bool):
        super().__init__()
        self.emb_dim = emb_dim
        self.n_experts = n_experts
        self.n_experts_per_token = n_experts_per_token
        self.moe_norm_topk_prob = moe_norm_topk_prob
        self.proj = nn.Linear(emb_dim, n_experts, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape: (num_tokens, emb_dim)
        router_logits = self.proj(x)
        # softmax over all experts first, then select top-k (matches HF 5.x)
        router_probs = F.softmax(router_logits, dtype=torch.float32, dim=-1)
        # values/indices shape: (num_tokens, n_experts_per_token)
        values, indices = torch.topk(router_probs, self.n_experts_per_token, dim=-1)
        if self.moe_norm_topk_prob:
            values = values / values.sum(dim=-1, keepdim=True)
        return values, indices


class SparseMoEBlock(nn.Module):
    def __init__(self, emb_dim: int, n_experts: int, n_experts_per_token: int, moe_hidden_dim: int, moe_norm_topk_prob: bool):
        super().__init__()
        self.emb_dim = emb_dim
        self.n_experts = n_experts
        self.n_experts_per_token = n_experts_per_token
        self.moe_norm_topk_prob = moe_norm_topk_prob
        # fused 3D expert weights (matches HF 5.x Qwen3MoeExperts)
        self.gate_up_proj = nn.Parameter(torch.empty(n_experts, 2 * moe_hidden_dim, emb_dim))
        self.down_proj = nn.Parameter(torch.empty(n_experts, emb_dim, moe_hidden_dim))
        self.router = MoERouter(self.emb_dim, self.n_experts, self.n_experts_per_token, self.moe_norm_topk_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape: (batch, seq_len, emb_dim)
        batch, seq_len, emb_dim = x.shape
        hidden_states = x.view(-1, emb_dim)  # (num_tokens, emb_dim)

        routing_weights, selected_experts = self.router(hidden_states)
        # routing_weights: (num_tokens, n_experts_per_token)  float32
        # selected_experts: (num_tokens, n_experts_per_token)

        final_hidden_states = torch.zeros_like(hidden_states)  # (num_tokens, emb_dim)

        for expert_idx in selected_experts.unique():
            token_idx, top_k_pos = torch.where(selected_experts == expert_idx)
            current_state = hidden_states[token_idx]

            # fused gate+up projection, then chunk (matches HF 5.x computation order)
            gate, up = F.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = F.silu(gate) * up
            current_hidden_states = F.linear(current_hidden_states, self.down_proj[expert_idx])

            # weight by routing score and accumulate
            current_hidden_states = current_hidden_states * routing_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states.view(batch, seq_len, emb_dim)


class MoETransformersBlock(nn.Module):
    def __init__(self, emb_dim: int, n_heads: int, n_kv_groups: int, head_dim: int, n_experts: int, n_experts_per_token: int, moe_hidden_dim: int, moe_norm_topk_prob: bool):
        super().__init__()
        self.emb_dim = emb_dim
        self.n_heads = n_heads
        self.n_kv_groups = n_kv_groups
        self.head_dim = head_dim
        self.n_experts = n_experts
        self.n_experts_per_token = n_experts_per_token
        self.moe_hidden_dim = moe_hidden_dim
        self.moe_norm_topk_prob = moe_norm_topk_prob

        self.norm1 = RMSNorm(emb_dim)
        self.norm2 = RMSNorm(emb_dim)

        self.attn = GroupQueryAttention(self.emb_dim, self.n_heads, self.n_kv_groups, self.head_dim)
        self.moe_ffn = SparseMoEBlock(self.emb_dim, self.n_experts, self.n_experts_per_token, self.moe_hidden_dim, self.moe_norm_topk_prob)
    
    def forward(self, x, rope, position_ids, kv_cache=None, attn_mask=None):
        residual = x
        x = self.norm1(x)
        x, new_kv_cache = self.attn(x, rope, position_ids, kv_cache, attn_mask)
        x = x + residual

        residual = x
        x = self.norm2(x)
        x = self.moe_ffn(x)
        x = x + residual

        return x, new_kv_cache