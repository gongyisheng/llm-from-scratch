import torch
import torch.distributed as dist


def init_process_group(rank: int, world_size: int, backend: str = "gloo"):
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size,
                            init_method="tcp://127.0.0.1:29500")


def get_rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def all_reduce(tensor: torch.Tensor) -> torch.Tensor:
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def destroy_process_group():
    if dist.is_initialized():
        dist.destroy_process_group()
