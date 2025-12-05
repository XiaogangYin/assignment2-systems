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
        handles = []
        for p in list(self.module.parameters())[::-1]:
            if p.grad is not None:
                handle = dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, async_op=True)
                handles.append(handle)

        for handle in handles:
            handle.wait()
        handles.clear()
        for p in self.module.parameters():
            if p.grad is not None:
                p.grad.div_(self.world_size)


class DDPBucketed:
    """
    Implement a Python class to handle distributed data parallel training, using gradient bucketing to
    improve communication efficiency. The class should wrap an arbitrary input PyTorch nn.Module and
    take care of broadcasting the weights before training (so all ranks have the same initial parameters) and
    issuing bucketed communication calls for gradient averaging.
    """
    def __init__(self, module: torch.nn.Module, bucket_size_mb: float):
        """
        Given an instantiated PyTorch nn.Module to be parallelized, construct a DDP container
        that will handle gradient synchronization
        across ranks. Gradient synchronization should be bucketed, with each bucket holding
        at most bucket_size_mb of parameters
        """
        self.module = module
        for param in self.module.parameters():
            dist.broadcast(param, 0)
        self.bucket_size_bytes = int(bucket_size_mb * 1024 * 1024)
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
        handles = []
        buckets = []
        param_buckets = []
        current_grad_bucket = []  # 当前桶
        current_param_bucket = []
        current_bucket_bytes = 0  # 当

        def calculate_tensor_bytes(tensor: torch.Tensor) -> int:
            return tensor.numel() * tensor.element_size()

        for p in self.module.parameters():
            if not p.requires_grad:
                continue
            bytes = calculate_tensor_bytes(p.grad)
            if bytes > self.bucket_size_bytes:
                if current_grad_bucket:
                    z = torch._utils._flatten_dense_tensors(current_grad_bucket)
                    buckets.append(z)
                    param_buckets.append(current_param_bucket)
                    handle = dist.all_reduce(z, op=dist.ReduceOp.SUM, async_op=True)
                    handles.append(handle)
                    current_grad_bucket = []
                    current_param_bucket = []
                    current_bucket_bytes = 0
                buckets.append(p.grad)
                param_buckets.append([p])
                handle = dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, async_op=True)
                handles.append(handle)
                continue

            if bytes + current_bucket_bytes > self.bucket_size_bytes:
                z = torch._utils._flatten_dense_tensors(current_grad_bucket)
                buckets.append(z)
                param_buckets.append(current_param_bucket)
                handle = dist.all_reduce(z, op=dist.ReduceOp.SUM, async_op=True)
                handles.append(handle)
                current_grad_bucket = [p.grad]
                current_param_bucket = [p]
                current_bucket_bytes = bytes
            else:
                current_bucket_bytes += bytes
                current_param_bucket.append(p)
                current_grad_bucket.append(p.grad)

        if current_grad_bucket:
            z = torch._utils._flatten_dense_tensors(current_grad_bucket)
            buckets.append(z)
            param_buckets.append(current_param_bucket)
            handle = dist.all_reduce(z, op=dist.ReduceOp.SUM, async_op=True)
            handles.append(handle)


        for handle in handles:
            handle.wait()
        handles.clear()

        for param_bucket, z in zip(param_buckets, buckets):
            z.div_(self.world_size)
            if len(param_bucket) == 1:
                continue
            new_grads = torch._utils._unflatten_dense_tensors(z, param_bucket)
            for g, p in zip(new_grads, param_bucket):
                p.grad = g

