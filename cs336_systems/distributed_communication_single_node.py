import os
import sys
import timeit
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def distributed_demo(rank, world_size, data_size, q):
    setup(rank, world_size)
    data = torch.randn(data_size//world_size)
    before_time = timeit.default_timer()
    #print(f"rank {rank} data (before all-reduce) time: {before_time}")
    dist.all_reduce(data, async_op=False)
    all_reduce_runtime = timeit.default_timer() - before_time
    q.put((rank, all_reduce_runtime))
    #print(f"rank {rank} data (after all-reduce), time: {timeit.default_timer()}, spend time: {all_reduce_runtime}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python distributed_communication.py world_size data_size")
        exit(1)
    world_size = int(sys.argv[1])
    data_size = int(sys.argv[2])
    q = mp.Queue()
    ctx = mp.spawn(fn=distributed_demo, args=(world_size, data_size, q, ), nprocs=world_size, join=False)

    all_results = []
    for _ in range(world_size):
        rank, result = q.get()
        all_results.append( (rank, result) )
    ctx.join()  # 这里的 join() 是等待所有子进程结束，和 spawn 的 join 参数不同

    # 5. 汇总结果
    #print("\n所有子进程结果汇总：")
    #for rank, result in sorted(all_results):
    #    print(f"rank {rank}：{result:.4f}")
    total = sum([res for _, res in all_results]) / world_size
    print(f"{data_size//1000000}M {world_size} {total:.4f}")
