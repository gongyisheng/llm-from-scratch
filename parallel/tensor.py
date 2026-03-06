import torch
import torch.nn as nn
import torch.nn.functional as F

from parallel.comm import all_reduce, get_rank, get_world_size


class ColumnParallelLinear(nn.Module):
    """Linear layer with output dimension split across ranks.

    weight shape: (out_features // world_size, in_features)
    No communication needed in forward.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        world_size = get_world_size()
        # nn.Linear stores tensor as (out_feat, in_feat), will shard by dim=0
        self.weight = nn.Parameter(torch.empty(out_features // world_size, in_features))
        self.bias = nn.Parameter(torch.empty(out_features // world_size,)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)

class RowParallelLinear(nn.Module):
    """Linear layer with input dimension split across ranks.

    weight shape: (out_features, in_features // world_size)
    all_reduce after matmul when world_size > 1.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        world_size = get_world_size()
        # nn.Linear stores tensor as (out_feat, in_feat), will shard by dim=1
        self.weight = nn.Parameter(torch.empty(out_features, in_features // world_size))
        self.bias = nn.Parameter(torch.empty(out_features,)) if bias else None

    def forward(self, x):
        # Bias must be added AFTER all_reduce, not before.
        # Each rank holds the same (replicated) bias, so adding it before
        # all_reduce would sum it world_size times.
        out = F.linear(x, self.weight)
        out = all_reduce(out)
        if self.bias is not None:
            out = out + self.bias
        return out


class ParallelEmbedding(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.rank = get_rank()
        self.world_size = get_world_size()
        self.embd_per_rank = in_features // self.world_size
        self.start = self.rank * self.embd_per_rank
        self.end = (self.rank + 1) * self.embd_per_rank
        self.weight = nn.Parameter(torch.empty(self.embd_per_rank, out_features))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len)
        mask = (x >= self.start) & (x < self.end)
        local_ids = (x - self.start).clamp(0, self.embd_per_rank - 1)
        output = F.embedding(local_ids, self.weight) # (batch, seq_len, emb_dim)
        output = output * mask.unsqueeze(-1) # must unsqueeze to match
        output = all_reduce(output)
        return output


def shard_tensor(tensor: torch.Tensor, dim: int):
    world_size = get_world_size()
    size = tensor.shape[dim]
    shard_size = size // world_size
    start = get_rank() * shard_size
    return tensor.narrow(dim, start, shard_size).contiguous()