from typing import Type

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim

class DDPIndividualParameters:
    """
    Implement a Python class to handle distributed data parallel training. The class should wrap
    an arbitrary PyTorch nn.Module and take care of broadcasting the weights before training
    (so all ranks have the same initial parameters) and issuing communication calls for gradient averaging
    """
    def __init__(self, module: torch.nn.Module):
        """
        Given an instantiated PyTorch nn.Module to be parallelized,
        construct a DDP container that will handle gradient synchronization across ranks.
        """
        self.module = module
        for param in self.module.parameters():
            dist.broadcast(param, 0)
        self.world_size = dist.get_world_size()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def named_parameters(self):
        return self.module.named_parameters()

    def parameters(self):
        return self.module.parameters()

    def forward(self, *inputs, **kwargs):
        """
        Calls the wrapped module’s forward() method with the provided positional and keyword arguments.
        """
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        """
        When called, wait for asynchronous communication calls to be queued on GPU.
        """

        for p in self.module.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, async_op=False)
                p.grad.div_(self.world_size)
