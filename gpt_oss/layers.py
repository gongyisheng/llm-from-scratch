import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, emb_dim: int, eps: float = 1e-6):
        super().__init__()
        self.emb_dim = emb_dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(self.emb_dim))
    
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
    def __init__(self, emb_dim: int, head_dim: int, n_heads: int, n_kv_groups: int, layer_idx: int, sliding_window: int):
        super().__init__()
        self.emb_dim = emb_dim
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.n_kv_groups = n_kv_groups
        self.layer_idx = layer_idx
        self.sliding_window = sliding_window
        self.group_size = n_heads // n_kv_groups

        self.sinks = nn.Parameter(torch.zeros(1, n_heads, 1, 1))
        self.q_proj = nn.Linear(self.emb_dim, self.head_dim * self.n_heads, bias=True)
        self.k_proj = nn.Linear(self.emb_dim, self.head_dim * self.n_kv_groups, bias=True)
        self.v_proj = nn.Linear(self.emb_dim, self.head_dim * self.n_kv_groups, bias=True)
        self.o_proj = nn.Linear(self.head_dim * self.n_heads, self.emb_dim, bias=True)
    
    def forward(self, x: torch.Tensor, rope: RoPE, position_ids: torch.Tensor, kv_cache=None, attn_mask=None) -> torch.Tensor:
        # x.shape: (batch, seq_len, emb_dim)
        batch, seq_len, _ = x.shape
        
        # reshape
        Q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch, seq_len, self.n_kv_groups, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch, seq_len, self.n_kv_groups, self.head_dim).transpose(1, 2)

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
        elif kv_cache is None:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
            scores.masked_fill_(mask, float("-inf"))
        
        # sliding windown mask on even layer
        _, _, full_seq_len, _ = scores.shape
        if self.layer_idx % 2 == 0 and full_seq_len > self.sliding_window:
            mask = torch.tril(torch.ones(full_seq_len, full_seq_len, device=x.device), diagonal=-self.sliding_window).bool()
            scores.masked_fill_(mask, float("-inf"))
        
        # attn sink
        # 1, n_head, 1, 1
        # batch, n_head, seq_len, seq_len
        sinks = self.sinks.expand(batch, -1, full_seq_len, -1)
        scores = torch.concat([sinks, scores], dim=-1)
        attn = F.softmax(scores, dim=-1, dtype=x.dtype)
        attn = attn[:, :, :, 1:]

        context = attn @ V
        context = context.transpose(1, 2).reshape(batch, seq_len, -1)
        output = self.o_proj(context)
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


class MoEGate(nn.Module):
    def __init__(self, emb_dim: int):
        super().__init__()
    
    def forward(self, x: torch.Tensor):
        pass