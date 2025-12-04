import logging
from copy import deepcopy
from typing import Type

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim


from tests.common import (
    FIXTURES_PATH,
    ToyModel,
    ToyModelWithTiedWeights,
    _cleanup_process_group,
    _setup_process_group,
    validate_ddp_net_equivalence,
)

from cs336_basics.train_utils import *

def setup(rank, world_size):
    #os.environ['GLOO_SOCKET_IFNAME'] = 'en0'
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def naive_ddp(rank: int, world_size: int, model_class: Type[torch.nn.Module]):
    # Use gloo backend for CPU
    device = _setup_process_group(rank=rank, world_size=world_size, backend="gloo")
    # Execute barrier prior to running test to ensure that every process
    # has finished initialization and that the following test
    # immediately exiting due to a skip doesn't cause flakiness.
    dist.barrier()

    # Seed to ensure that ranks are initialized with one initial models.
    torch.manual_seed(0)

    # Create a toy model and move it to the proper device.
    # This is our non-parallel baseline.
    model = model_class().to(device)
    model.train()

    # Load the dataset from disk, so we can ensure that every rank has the same
    # overall pool of data.
    # Shape: (20, 10)
    all_x = torch.load(FIXTURES_PATH / "ddp_test_data.pt")
    # Shape: (20, 5)
    all_y = torch.load(FIXTURES_PATH / "ddp_test_labels.pt")
    torch.manual_seed(42 + rank)
    shuffle_idxs = torch.randperm(all_x.size(0))
    all_x = all_x[shuffle_idxs]
    all_y = all_y[shuffle_idxs]

    local_bs = int(all_y.size(0) / world_size)

    loss_fn = nn.MSELoss()

    # Optimizer for the DDP model
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    for i in range(5):
        torch.manual_seed(rank + i * local_bs)
        torch.randint(all_x.size(0), (local_bs,))
        shuffle_idxs = torch.randperm(all_x.size(0))[:local_bs]
        x = all_x[shuffle_idxs]
        y = all_y[shuffle_idxs]
        outputs = model(x)
        loss = loss_fn(outputs, y)

        model.zero_grad()
        loss.backward()
        for p in model.parameters():
            dist.all_reduce(p.grad,  op=dist.ReduceOp.SUM, async_op=False)
            p.grad.div_(world_size)
        print(i, "loss:", loss.item())
        optimizer.step()
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 4
    mp.spawn(fn=naive_ddp, args=(world_size, ToyModelWithTiedWeights,), nprocs=world_size, join=True)