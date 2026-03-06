import torch
import torch.nn as nn
import torch.nn.functional as F

from parallel.comm import all_reduce


class ColumnParallelLinear(nn.Module):
    """Linear layer with output dimension split across ranks.

    weight shape: (out_features // world_size, in_features)
    No communication needed in forward.
    """
    def __init__(self, in_features: int, out_features: int, world_size: int = 1, bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features // world_size, in_features))
        self.bias = nn.Parameter(torch.empty(out_features // world_size,)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)

class RowParallelLinear(nn.Module):
    """Linear layer with input dimension split across ranks.

    weight shape: (out_features, in_features // world_size)
    all_reduce after matmul when world_size > 1.
    """
    def __init__(self, in_features: int, out_features: int, world_size: int = 1, bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features // world_size))
        self.bias = nn.Parameter(torch.empty(out_features,)) if bias else None

    def forward(self, x):
        out = F.linear(x, self.weight, self.bias)
        out = all_reduce(out)
        return out
