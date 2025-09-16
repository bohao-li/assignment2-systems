import os
import time

import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def setup(rank, world_size, backend, device_type):
    """
    Initializes the distributed process group.
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    if device_type == "cuda":
        torch.cuda.set_device(rank)


def cleanup():
    """
    Destroys the process group to release resources.
    """
    dist.destroy_process_group()


def run_benchmark(rank, world_size, backend, data_size_mb, device_type, results_list):
    """
    Runs a simple distributed all-reduce operation benchmark and appends results.
    """
    setup(rank, world_size, backend, device_type)

    # Convert MB to number of float32 elements
    num_elements = int(data_size_mb * 1024 * 1024 / 4)  # 4 bytes per float32

    if device_type == "cuda":
        device = torch.device(f"cuda:{rank}")
        data = torch.randint(0, 10, (num_elements,), dtype=torch.float32, device=device)
    else:
        device = torch.device("cpu")
        data = torch.randint(0, 10, (num_elements,), dtype=torch.float32, device=device)

    num_iterations = 10
    num_warmup = 3

    # Warm-up iterations
    for _ in range(num_warmup):
        dist.all_reduce(data, op=dist.ReduceOp.SUM, async_op=False)

    # Synchronize and start timer
    dist.barrier()
    start_time = time.time()

    # Benchmark loop
    for _ in range(num_iterations):
        dist.all_reduce(data, op=dist.ReduceOp.SUM, async_op=False)

    # Synchronize and stop timer
    dist.barrier()
    end_time = time.time()

    # Calculate average time per iteration
    avg_time = (end_time - start_time) / num_iterations

    if rank == 0:
        results_list.append(
            {
                "Backend": backend,
                "Device": device_type,
                "Processes": world_size,
                "Data Size (MB)": data_size_mb,
                "Average Time (s)": avg_time,
            }
        )
        print(
            f"Backend: {backend}, Device: {device_type}, Processes: {world_size}, "
            f"Data Size: {data_size_mb}MB, Avg Time: {avg_time:.6f}s"
        )

    cleanup()


if __name__ == "__main__":
    # Updated to reflect a maximum of 4 GPUs
    WORLD_SIZES = [2, 4]
    DATA_SIZES_MB = [1, 10, 100, 1024]  # 1024MB = 1GB

    # Use multiprocessing.Manager to share a list for collecting results
    manager = mp.Manager()
    results_list = manager.list()

    # Benchmarking Gloo (CPU)
    print("--- Gloo (CPU) Benchmark ---")
    for world_size in WORLD_SIZES:
        for data_size_mb in DATA_SIZES_MB:
            mp.spawn(
                fn=run_benchmark,
                args=(world_size, "gloo", data_size_mb, "cpu", results_list),
                nprocs=world_size,
                join=True,
            )

    # Benchmarking NCCL (GPU)
    if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
        print("\n--- NCCL (GPU) Benchmark ---")
        for world_size in WORLD_SIZES:
            if torch.cuda.device_count() >= world_size:
                for data_size_mb in DATA_SIZES_MB:
                    mp.spawn(
                        fn=run_benchmark,
                        args=(world_size, "nccl", data_size_mb, "cuda", results_list),
                        nprocs=world_size,
                        join=True,
                    )
            else:
                print(
                    f"Skipping NCCL benchmark for world_size={world_size} due to insufficient GPUs."
                )
    else:
        print(
            "\nSkipping all NCCL benchmarks: CUDA not available or fewer than 2 GPUs found."
        )

    # Convert the shared list to a pandas DataFrame and save to CSV
    results_df = pd.DataFrame(list(results_list))
    results_df.to_csv("all_reduce_benchmark_results.csv", index=False)
    print("\nBenchmark results saved to 'all_reduce_benchmark_results.csv'.")
