import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.to(torch.float32)
        rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt()
        output = x / (rms + self.eps) * self.weight
        output = output.to(dtype)
        return output


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
        cos = self.cos[position_ids]
        sin = self.sin[position_ids]

        cos = cos[:, None, :, :] # (batch, 1, seq_len, head_dim)
        sin = sin[:, None, :, :]

        x1 = x[..., :self.head_dim // 2]
        x2 = x[..., self.head_dim // 2:]

        rotated = torch.concat([-x2, x1], dim=-1)
        output = x*cos + rotated*sin
        return output


class GroupQueryAttention(nn.Module):
    def __init__(self, emb_dim: int, n_heads: int, n_kv_groups: int, head_dim: int):
        super().__init__()
        self.emb_dim = emb_dim
        self.n_heads = n_heads
        self.n_kv_groups = n_kv_groups
        self.head_dim = head_dim

        self.q_proj = nn.Linear(self.emb_dim, self.head_dim * self.n_heads, bias=False)
        self.k_proj = nn.Linear(self.emb_dim, self.head_dim * self.n_kv_groups, bias=False)
        self.v_proj = nn.Linear(self.emb_dim, self.head_dim * self.n_kv_groups, bias=False)
        self.o_proj = nn.Linear(self.head_dim * self.n_heads, self.emb_dim, bias=False)

        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
    
    def forward(self, x, rope, position_ids, kv_cache=None, attn_mask=None) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # x.shape: (batch, seq, emb_dim)
        batch, seq_len, _ = x.shape
    
        Q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1,2)
        K = self.k_proj(x).view(batch, seq_len, self.n_kv_groups, self.head_dim).transpose(1,2)
        V = self.v_proj(x).view(batch, seq_len, self.n_kv_groups, self.head_dim).transpose(1,2)

        Q = self.q_norm(Q.reshape(-1, seq_len, self.head_dim))
        Q = Q.view(batch, self.n_heads, seq_len, self.head_dim)

        K = self.k_norm(K.reshape(-1, seq_len, self.head_dim))
        K = K.view(batch, self.n_kv_groups, seq_len, self.head_dim)

        Q = rope(Q, position_ids)
        K = rope(K, position_ids)

        if kv_cache is not None:
            past_k, past_v = kv_cache
            # decode, each step seq_len = 1, concat at seq_len dim
            K = torch.concat([past_k, K], dim=2)
            V = torch.concat([past_v, V], dim=2)
        new_kv_cache = (K, V)

        group_size = self.n_heads // self.n_kv_groups
        K = K.repeat_interleave(group_size, dim=1) # repeat at n_kv_groups dim
        V = V.repeat_interleave(group_size, dim=1) # repeat at n_kv_groups dim

        scores = Q @ K.transpose(-2, -1) / (self.head_dim ** 0.5) # (batch, n_heads, seq_len, total_seq_len)
        if attn_mask is not None:
            # prefill
            scores += attn_mask
        elif kv_cache is None:
            # single sequence
            mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
            scores.masked_fill_(mask, float("-inf"))  # fill -inf to make it's 0 after softmax
        attn = F.softmax(scores, dim=-1, dtype=V.dtype)
        context = attn @ V

        context = context.transpose(1,2).reshape(batch, seq_len, -1)
        output = self.o_proj(context)
        return output, new_kv_cache


class SwiGLUFFN(nn.Module):
    def __init__(self, emb_dim: int, moe_hidden_dim: int):
        super().__init__()
        self.emb_dim = emb_dim
        self.moe_hidden_dim = moe_hidden_dim
        self.gate_proj = nn.Linear(self.emb_dim, self.moe_hidden_dim, bias=False)
        self.up_proj = nn.Linear(self.emb_dim, self.moe_hidden_dim, bias=False)
        self.down_proj = nn.Linear(self.moe_hidden_dim, self.emb_dim, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape: (batch, seq_len, emb_dim)
        gate = F.silu(self.gate_proj(x))                # gate.shape: (batch, seq_len, moe_hidden_dim)
        output = self.down_proj(gate * self.up_proj(x)) # output.shape: (batch, seq_len, emb_dim)
        return output


class MoEGate(nn.Module):
    def __init__(self, emb_dim: int, n_experts: int, n_experts_per_token: int, moe_norm_topk_prob: bool):
        super().__init__()
        self.emb_dim = emb_dim
        self.n_experts = n_experts
        self.n_experts_per_token = n_experts_per_token
        self.moe_norm_topk_prob = moe_norm_topk_prob
        self.proj = nn.Linear(emb_dim, n_experts, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape: (num_tokens, emb_dim)
        router_scores = self.proj(x)
        # values/indices shape: (batch, seq_len, n_experts_per_token)
        values, indices = torch.topk(router_scores, self.n_experts_per_token)
        if self.moe_norm_topk_prob:
            values = F.softmax(values, dim=-1)
        return values, indices


class SparseMoEBlock(nn.Module):
    def __init__(self, emb_dim: int, n_experts: int, n_experts_per_token: int, moe_hidden_dim: int, moe_norm_topk_prob: bool):
        super().__init__()
        self.emb_dim = emb_dim
        self.n_experts = n_experts
        self.n_experts_per_token = n_experts_per_token
        self.moe_hidden_dim = moe_hidden_dim
        self.moe_norm_topk_prob = moe_norm_topk_prob
        self.experts = nn.ModuleList([
            SwiGLUFFN(self.emb_dim, self.moe_hidden_dim) for _ in range(n_experts)
        ])
        self.gate = MoEGate(self.emb_dim, self.n_experts, self.n_experts_per_token, self.moe_norm_topk_prob)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape: (batch, seq_len, emb_dim)
        batch, seq_len, emb_dim = x.shape
        hidden_states = x.view(-1, emb_dim)  # (num_tokens, emb_dim)

        routing_weights, selected_experts = self.gate(hidden_states)
        # routing_weights: (num_tokens, n_experts_per_token)
        # selected_experts: (num_tokens, n_experts_per_token)

        output = torch.zeros_like(hidden_states)  # (num_tokens, emb_dim)

        for expert_idx in range(self.n_experts):
            # find which tokens selected this expert
            token_idx, slot_idx = torch.where(selected_experts == expert_idx)
            if token_idx.numel() == 0:
                continue

            # run selected tokens through this expert
            expert_output = self.experts[expert_idx](hidden_states[token_idx])

            # weight by routing score and accumulate
            weights = routing_weights[token_idx, slot_idx].unsqueeze(-1)
            output[token_idx] += expert_output * weights

        return output.view(batch, seq_len, emb_dim)


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