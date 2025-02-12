import torch
import torch.distributed as dist
import time
import os

# 初始化分布式环境
def init_distributed_backend(backend='nccl'):
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(f"Rank {rank}/{world_size} initialized.")
    return rank, world_size

# 测量AllReduce操作的性能
def test_allreduce(rank, world_size, tensor_size):
    tensor = torch.ones(tensor_size).to(rank)

    # 同步AllReduce
    dist.barrier()  # 确保所有进程同步
    start_time = time.time()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    end_time = time.time()

    # 计算并输出AllReduce的时间
    elapsed_time = end_time - start_time
    print(f"Rank {rank}: AllReduce took {elapsed_time:.6f} seconds.")

# 测量AllGather操作的性能
def test_allgather(rank, world_size, tensor_size):
    tensor = torch.ones(tensor_size).to(rank)
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]

    # 同步AllGather
    dist.barrier()
    start_time = time.time()
    dist.all_gather(gathered_tensors, tensor)
    end_time = time.time()

    # 计算并输出AllGather的时间
    elapsed_time = end_time - start_time
    print(f"Rank {rank}: AllGather took {elapsed_time:.6f} seconds.")

def main():
    # 设置环境变量（可根据实际情况调整）
    backend = 'nccl'  # 'nccl'适用于GPU，'gloo'适用于CPU
    tensor_size = 1000000  # 调整为所需的张量大小

    # 初始化分布式环境
    rank, world_size = init_distributed_backend(backend)

    # 测试AllReduce
    test_allreduce(rank, world_size, tensor_size)

    # 测试AllGather
    test_allgather(rank, world_size, tensor_size)

    # 清理分布式环境
    dist.destroy_process_group()

if __name__ == "__main__":
    main()