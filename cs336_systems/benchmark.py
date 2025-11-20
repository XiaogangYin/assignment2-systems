from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy

import time
import timeit
from typing import Callable
import torch


def mean(x: list[float]) -> float:
    return sum(x) / len(x)

def get_device(index: int = 0) -> torch.device:
    """Try to use the GPU if possible, otherwise, use CPU."""
    if torch.cuda.is_available():
        return torch.device(f"cuda:{index}")
    else:
        return torch.device("cpu")

def run_operation2(dim: int, operation: Callable) -> Callable:
    # Setup: create two random dim x dim matrices
    x = torch.randn(dim, dim, device=get_device())
    y = torch.randn(dim, dim, device=get_device())
    # Return a function to perform the operation
    return lambda : operation(x, y)




def run_mlp(dim: int, num_layers: int, batch_size: int, num_steps: int) -> Callable:
    # Define a model (with random weights)
    model = MLP(dim, num_layers).to(get_device())

    # Define an input (random)
    x = torch.randn(batch_size, dim, device=get_device())

    def run():
        # Run the model `num_steps` times (note: no optimizer updates)
        for step in range(num_steps):
            # Forward
            y = model(x).mean()

            # Backward
            y.backward()

    return run

def run_gpt(d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        context_length: int = 256,
        vocab_size: int = 10000,
        rope_theta: float = 10000.,
        batch_size: int = 4,
        num_steps: int = 1,
        run_backward: bool = False,
    ) -> Callable:
    model = BasicsTransformerLM(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta).to(get_device())

    x = torch.randint(low=0, high=vocab_size, size=(batch_size, context_length), device=get_device())
    y = torch.randint(low=0, high=vocab_size, size=(batch_size, context_length), device=get_device())

    def run():
        for step in range(num_steps):
            # Forward
            output = model(x)
            if run_backward:
                loss = cross_entropy(output, y)
                loss.backward()

        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Wait for CUDA threads to finish (important!)

    return run


def benchmark(description: str, run: Callable, num_warmups: int = 5, num_trials: int = 10):
    """Benchmark `func` by running it `num_trials`, and return all the times."""
    # Warmup: first times might be slower due to compilation, things not cached.
    # Since we will run the kernel multiple times, the timing that matters is steady state.
    for _ in range(num_warmups):
        run()
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Wait for CUDA threads to finish (important!)

    # Time it for real now!, 必须设个globals, 不然不认识 run()
    exe_time = timeit.timeit(stmt="run()", setup="pass", number=num_trials, globals={"run": run})
    return exe_time / num_trials

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

    gpt_results = []
    for model_type in model_types:
        for run_backward in [False, True]:
            this_config = config[model_type].copy()
            this_config["run_backward"] = run_backward
            desc = f"gpt-{model_type}_backward-{run_backward}"
            result = benchmark(desc,
                               run_gpt(**this_config), num_trials=10)
            gpt_results.append((desc, result))  # @inspect gpt_results
        print(gpt_results)
    print(gpt_results)



if __name__ == "__main__":
    main()