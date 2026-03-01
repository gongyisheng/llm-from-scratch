import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, emb_dim: int, eps: float = 1e-6):
        super().__init__()
        self.emb_dim = emb_dim
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(self.emb_dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt()
        output = x / (rms + self.eps) * self.weight
        output = output.to(orig_dtype)
        return output


class RoPE(nn.Module):
    def __init__(self, 
            head_dim: int, 
            rope_base: int, 
            max_seq_len: int, 
            yarn_original_context_length: int,
            yarn_scaling_factor: float,
            yarn_beta_fast: int = 32,
            yarn_beta_slow: int = 1,
        ):
        super().__init__()
        self.head_dim = head_dim
        self.rope_base = rope_base
        self.max_seq_len = max_seq_len
        self.yarn_original_context_length = yarn_original_context_length
        self.yarn_scaling_factor = yarn_scaling_factor
        self.yarn_beta_fast = yarn_beta_fast
        self.yarn_beta_slow = yarn_beta_slow
        self._build_buffers()
    
    def _build_buffers(self):
        freqs = 1 / self.rope_base ** (torch.arange(0, self.head_dim, 2) / self.head_dim)

        # YaRN specific logic - slow down low freqs, make sure it doesn't break under extended context
        # transform freq to wave length, calc high/low threshold, recommend by paper 32/1
        wave_len = 2 * torch.pi / freqs
        low_threshold = self.yarn_original_context_length / self.yarn_beta_fast
        high_threshold = self.yarn_original_context_length / self.yarn_beta_slow

        # ramp (t), clip by 0/1, calculate new freqs
        # concentration to help model distinguish angle diff after interpolate (slow down)
        t = (wave_len - low_threshold) / (high_threshold - low_threshold)
        t = t.clamp(0, 1)
        effective_freqs = freqs * (1 - t) + (freqs / self.yarn_scaling_factor) * t
        # concentration is an empirical formula
        concentration = 0.1 * torch.log(torch.tensor(self.yarn_scaling_factor)) + 1.0

        positions = torch.arange(self.max_seq_len)
        angles = positions[:, None] * effective_freqs[None, :]
        angles = angles * concentration

        angles = torch.cat([angles, angles], dim=-1)
        self.register_buffer("sin", torch.sin(angles))
        self.register_buffer("cos", torch.cos(angles))
    
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        sin = self.sin[position_ids]
        cos = self.cos[position_ids] # batch, seq_len, head_dim

        sin = sin[:, None, :, :]
        cos = cos[:, None, :, :]

        x1 = x[..., :self.head_dim // 2] # batch, n_head, seq_len, head_dim // 2
        x2 = x[..., self.head_dim // 2:]

        rotated = torch.concat([-x2, x1], dim=-1)
        output = x * cos + rotated * sin
        return output


class GroupQueryAttention(nn.Module):
    def __init__(self, emb_dim: int, n_heads: int, n_kv_groups: int, head_dim: int, layer_idx: int, sliding_window: int):
        super().__init__()
        self.emb_dim = emb_dim
        self.n_heads = n_heads
        self.n_kv_groups = n_kv_groups
        self.head_dim = head_dim
        self.layer_idx = layer_idx
        self.sliding_window = sliding_window
        self.group_size = n_heads // n_kv_groups

        self.sinks = nn.Parameter(torch.empty(1, n_heads, 1, 1))
        self.qkv_proj = nn.Linear(self.emb_dim, (self.n_heads + 2 * self.n_kv_groups) * self.head_dim, bias=True)
        self.out = nn.Linear(self.head_dim * self.n_heads, self.emb_dim, bias=True)
    
    def forward(self, x: torch.Tensor, rope: RoPE, position_ids: torch.Tensor, kv_cache=None, attn_mask=None) -> torch.Tensor:
        # x.shape: (batch, seq_len, emb_dim)
        batch, seq_len, _ = x.shape
        
        # get fused qkv, split by head
        QKV = self.qkv_proj(x).view(batch, seq_len, (self.n_heads + 2*self.n_kv_groups), self.head_dim)
        
        # split QKV, reshape
        Q, K, V = torch.split(QKV, [self.n_heads, self.n_kv_groups, self.n_kv_groups], dim=2)

        Q = Q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch, seq_len, self.n_kv_groups, self.head_dim).transpose(1, 2)
        V = V.view(batch, seq_len, self.n_kv_groups, self.head_dim).transpose(1, 2)

        # apply rope
        Q = rope(Q, position_ids)
        K = rope(K, position_ids)

        if kv_cache is not None:
            past_k, past_v = kv_cache
            K = torch.concat([past_k, K], dim=2)
            V = torch.concat([past_v, V], dim=2)
        new_kv_cache = (K, V)

        K = K.repeat_interleave(self.group_size, dim=1)
        V = V.repeat_interleave(self.group_size, dim=1)

        scores = Q @ K.transpose(-2, -1) / self.head_dim ** 0.5
        if attn_mask is not None:
            scores += attn_mask
        
        # sliding windown mask on even layer
        _, _, q_len, kv_len = scores.shape
        if self.layer_idx % 2 == 0 and kv_len > self.sliding_window:
            mask = torch.tril(torch.ones(q_len, kv_len, device=x.device), diagonal=-self.sliding_window).bool()
            scores.masked_fill_(mask, float("-inf"))
        
        # attn sink
        # 1, n_head, 1, 1
        # batch, n_head, seq_len, seq_len
        sinks = self.sinks.expand(batch, -1, q_len, -1)
        scores = torch.concat([sinks, scores], dim=-1)
        attn = F.softmax(scores, dim=-1, dtype=x.dtype)
        attn = attn[:, :, :, 1:]

        context = attn @ V
        context = context.transpose(1, 2).reshape(batch, seq_len, -1)
        output = self.out(context)
        return output, new_kv_cache


class SwiGLUFFN(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_gate = x[..., ::2] # gate
        x_up = x[..., 1::2] # up
        # gelu approximation, x * sigmoid(1.702x)
        # use bias = 1 for x_up: help with x_up to learn during initialization
        # avoid both x_gate and x_up are almost 0 and cause "dead zero"
        output = x_gate.clamp(-7, 7) * torch.sigmoid(1.702 * x_gate) * (x_up.clamp(-7, 7) + 1)
        return output


class MoERouter(nn.Module):
    def __init__(self, emb_dim: int, n_experts: int, n_experts_per_token: int, moe_norm_topk_prob: bool):
        super().__init__()
        self.emb_dim = emb_dim
        self.n_experts = n_experts
        self.n_experts_per_token = n_experts_per_token
        self.moe_norm_topk_prob = moe_norm_topk_prob
        self.proj = nn.Linear(self.emb_dim, self.n_experts, bias=True)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x.shape: (batch, seq_len, emb_dim)
        router_scores = self.proj(x)
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

        self.router = MoERouter(self.emb_dim, self.n_experts, self.n_experts_per_token, self.moe_norm_topk_prob)
        self.activation = SwiGLUFFN()
        self.gate_up_proj_weight = nn.Parameter(torch.empty(self.n_experts, self.moe_hidden_dim * 2, self.emb_dim)) # fused layer
        self.gate_up_proj_bias = nn.Parameter(torch.empty(self.n_experts, self.moe_hidden_dim * 2))
        self.down_proj_weight = nn.Parameter(torch.empty(self.n_experts, self.emb_dim, self.moe_hidden_dim))
        self.down_proj_bias = nn.Parameter(torch.empty(self.n_experts, self.emb_dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, emb_dim = x.shape
        hidden_state = x.view(-1, emb_dim)

        routing_weights, selected_experts = self.router(hidden_state)
        output = torch.zeros_like(hidden_state)

        for expert_idx in range(self.n_experts):
            token_idx, slot_idx = torch.where(selected_experts == expert_idx)
            if token_idx.numel() == 0:
                continue
            
            tokens = hidden_state[token_idx]
            expert_output = F.linear(tokens, self.gate_up_proj_weight[expert_idx], self.gate_up_proj_bias[expert_idx])
            expert_output = self.activation(expert_output)
            expert_output = F.linear(expert_output, self.down_proj_weight[expert_idx], self.down_proj_bias[expert_idx])
            
            weighted_output = expert_output * routing_weights[token_idx, slot_idx].unsqueeze(-1)
            output[token_idx] += weighted_output
        
        return output.view(batch, seq_len, emb_dim)


class MoETransformersBlock(nn.Module):
    def __init__(self, emb_dim: int, n_heads: int, n_kv_groups: int, head_dim: int, layer_idx: int, sliding_window: int, n_experts: int, n_experts_per_token: int, moe_hidden_dim: int, moe_norm_topk_prob: bool):
        super().__init__()
        self.emb_dim = emb_dim
        self.n_heads = n_heads
        self.n_kv_groups = n_kv_groups
        self.head_dim = head_dim
        self.layer_idx = layer_idx
        self.sliding_window = sliding_window
        self.n_experts = n_experts
        self.n_experts_per_token = n_experts_per_token
        self.moe_hidden_dim = moe_hidden_dim
        self.moe_norm_topk_prob = moe_norm_topk_prob

        self.norm1 = RMSNorm(emb_dim)
        self.norm2 = RMSNorm(emb_dim)

        self.attn = GroupQueryAttention(self.emb_dim, self.n_heads, self.n_kv_groups, self.head_dim, self.layer_idx, self.sliding_window)
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