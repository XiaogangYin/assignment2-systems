# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///

from collections.abc import Callable, Iterable
from typing import Optional, Union
import os
from ast import literal_eval
from typing import IO, Any, BinaryIO

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor

import math

__all__ = [
    "get_batch",
    "save_checkpoint",
    "load_checkpoint",
    "CfgNode",
]


def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int, 
    device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    Write a function that takes a numpy array x (integer array with token IDs), a
        batch_size, a context_length and a PyTorch device string (e.g., 'cpu' or 'cuda:0'), and returns
        a pair of tensors: the sampled input sequences and the corresponding next-token targets
    """
    n = dataset.shape[0]
    start_indices = np.random.randint(low=0, high=n-context_length, size=batch_size)
    # get batch_size x context_length indx array
    indices = start_indices[:, None] + np.arange(context_length) 
    next_indices = indices + 1
    return (torch.tensor(dataset[indices], dtype=torch.long, device=device),
            torch.tensor(dataset[next_indices], dtype=torch.long, device=device))


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    r"""
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html
    """
    state_dicts = {
        "model": model.state_dict(),
        "opt": optimizer.state_dict(),
        "iter": iteration
    }
    torch.save(state_dicts, out)

def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str = "auto"
) -> int:
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    state_dicts = torch.load(src, map_location=device)
    model.load_state_dict(state_dicts["model"])
    optimizer.load_state_dict(state_dicts["opt"])
    return state_dicts["iter"]

class CfgNode:
    """from karpathy minGPT"""
    """ a lightweight configuration class inspired by yacs """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __str__(self):
        return self._str_helper(0)

    def _str_helper(self, indent):
        """ need to have a helper to support nested indentation for pretty printing """
        parts = []
        for k, v in self.__dict__.items():
            if isinstance(v, CfgNode):
                parts.append("%s:\n" % k)
                parts.append(v._str_helper(indent + 1))
            else:
                parts.append("%s: %s\n" % (k, v))
        parts = [' ' * (indent * 4) + p for p in parts]
        return "".join(parts)

    def to_dict(self):
        """ return a dict representation of the config """
        return { k: v.to_dict() if isinstance(v, CfgNode) else v for k, v in self.__dict__.items() }

    def merge_from_dict(self, d):
        self.__dict__.update(d)

    def merge_from_args(self, args):
        """
        update the configuration from a list of strings that is expected
        to come from the command line, i.e. sys.argv[1:].

        The arguments are expected to be in the form of `--arg=value`, and
        the arg can use . to denote nested sub-attributes. Example:

        --model.n_layer=10 --trainer.batch_size=32
        """
        for arg in args:

            keyval = arg.split('=')
            assert len(keyval) == 2, "expecting each override arg to be of form --arg=value, got %s" % arg
            key, val = keyval # unpack

            # first translate val into a python object
            try:
                val = literal_eval(val)
                """
                need some explanation here.
                - if val is simply a string, literal_eval will throw a ValueError
                - if val represents a thing (like an 3, 3.14, [1,2,3], False, None, etc.) it will get created
                """
            except ValueError:
                pass

            # find the appropriate object to insert the attribute into
            assert key[:2] == '--'
            key = key[2:] # strip the '--'
            keys = key.split('.')
            obj = self
            for k in keys[:-1]:
                obj = getattr(obj, k)
            leaf_key = keys[-1]

            # ensure that this attribute exists
            assert hasattr(obj, leaf_key), f"{key} is not an attribute that exists in the config"

            # overwrite the attribute
            print("command line overwriting config attribute %s with %s" % (key, val))
            setattr(obj, leaf_key, val)
