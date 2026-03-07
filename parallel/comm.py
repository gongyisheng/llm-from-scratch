import os
import torch
import torch.distributed as dist

# rank: process num
# world_size: total process num
# backend: gloo(cpu tensor), nccl(gpu tensor)

def init_process_group(backend: str = "auto"):
    # torchrun sets RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if backend == "auto":
        # nccl requires one GPU per rank
        has_enough_gpus = (
            torch.cuda.is_available()
            and torch.cuda.device_count() >= world_size
        )
        backend = "nccl" if has_enough_gpus else "gloo"

    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    master_port = os.environ.get("MASTER_PORT", "29500")

    dist.init_process_group(
        backend=backend, rank=rank, world_size=world_size,
        init_method=f"tcp://{master_addr}:{master_port}",
    )

    # assign each rank to a GPU (wraps around when sharing GPUs)
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank % torch.cuda.device_count())
        # cublaslt is required for bfloat16/float16 GEMMs on H200 + CUDA 12.8+
        torch.backends.cuda.preferred_blas_library("cublaslt")


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
