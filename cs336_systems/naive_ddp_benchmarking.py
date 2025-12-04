from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy

import timeit
import time
from typing import Callable
import torch
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp

from tests.common import (
    FIXTURES_PATH,
    ToyModel,
    ToyModelWithTiedWeights,
    _cleanup_process_group,
    _setup_process_group,
    validate_ddp_net_equivalence,
)

def mean(x: list[float]) -> float:
    return sum(x) / len(x)

def get_device(index: int = 0) -> torch.device:
    """Try to use the GPU if possible, otherwise, use CPU."""
    if torch.cuda.is_available():
        return torch.device(f"cuda:{index}")
    else:
        return torch.device("cpu")

def run_gpt(rank: int,
            world_size: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        context_length: int = 256,
        vocab_size: int = 10000,
        rope_theta: float = 10000.,
        batch_size: int = 4,
        num_steps: int = 5,
    ) -> Callable:
    # Use gloo backend for CPU
    device = _setup_process_group(rank=rank, world_size=world_size, backend="gloo")
    # Execute barrier prior to running test to ensure that every process
    # has finished initialization and that the following test
    # immediately exiting due to a skip doesn't cause flakiness.
    dist.barrier()

    # Seed to ensure that ranks are initialized with one initial models.
    torch.manual_seed(0)

    model = BasicsTransformerLM(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta).to(get_device())

    x = torch.randint(low=0, high=vocab_size, size=(batch_size, context_length), device=get_device())
    y = torch.randint(low=0, high=vocab_size, size=(batch_size, context_length), device=get_device())
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    reduce_time = 0.
    total_time = 0.
    for step in range(num_steps):
        # Forward
        start = time.perf_counter()
        optimizer.zero_grad()
        output = model(x)
        loss = cross_entropy(output, y)
        loss.backward()
        start1 = time.perf_counter()
        for p in model.parameters():
            dist.all_reduce(p.grad,  op=dist.ReduceOp.SUM, async_op=False)
            p.grad.div_(world_size)
        reduce_time += time.perf_counter() - start1
        optimizer.step()
        total_time += time.perf_counter() - start
        print(f"total time: {total_time}, reduce time: {reduce_time}")

    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Wait for CUDA threads to finish (important!)
    dist.barrier()
    dist.destroy_process_group()
    print(f"{rank}, total time: {total_time/num_steps}, reduce time: {reduce_time/num_steps} ")

def run_gpt_wrapper(rank: int,
            world_size: int,
            config: dict,
            ):
    run_gpt(rank=rank, world_size=world_size, **config)

def main():
    # Run a larger model if GPU is available
    config = {
        "small": dict(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
        "medium": dict(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
        "large":dict(d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
        "xl":dict(d_model=1600, d_ff=6400, num_layers=48, num_heads=25),
        "2.7B":dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
    }
    if torch.cuda.is_available():
        model_types = config.keys() # @inspect dims
        model_types = list(model_types)[:3]
    else:
        model_types = ["small", "medium"]  # @inspect dims
        model_types = ["small"]

    for model_type in model_types:
        this_config = config[model_type].copy()
        world_size = 4
        mp.spawn(fn=run_gpt_wrapper, args=(world_size,this_config,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()