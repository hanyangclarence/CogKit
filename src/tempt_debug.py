# # filepath: minimal_test.py
# from accelerate import Accelerator
# print("Initializing Accelerator...")
# accelerator = Accelerator()
# print(f"Accelerator initialized. Device: {accelerator.device}, Process Index: {accelerator.process_index}, Num Processes: {accelerator.num_processes}")
# print("Minimal test finished.")

import torch
import torch.distributed as dist
import os
import argparse

def main():
    """Initializes the distributed environment and prints process information."""

    # torchrun automatically sets MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE, LOCAL_RANK
    # Initialize the process group
    # Use 'nccl' backend for GPU training, 'gloo' for CPU
    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    dist.init_process_group(backend=backend)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ['LOCAL_RANK'])

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        print(f"Rank {rank}/{world_size} (Local Rank {local_rank}) running on GPU: {torch.cuda.get_device_name(local_rank)}")
    else:
        device = torch.device("cpu")
        print(f"Rank {rank}/{world_size} (Local Rank {local_rank}) running on CPU.")

    # Example: Create a tensor unique to each process
    tensor = torch.tensor([rank], dtype=torch.float32, device=device)
    print(f"Rank {rank}: Initial tensor: {tensor.item()} on device {tensor.device}")

    # Example: Perform a simple collective operation (all-reduce)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print(f"Rank {rank}: Tensor after all_reduce sum: {tensor.item()}")

    # Clean up the process group
    dist.destroy_process_group()
    print(f"Rank {rank}: Distributed environment finished.")

if __name__ == "__main__":
    print("Starting minimal torchrun test...")
    main()
    print("Minimal torchrun test script finished.")