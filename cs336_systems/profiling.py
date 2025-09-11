# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

import argparse
import timeit
import traceback

import numpy as np

import torch
import torch.nn as nn

# from bohao.assignment2.cs336_basics.model import BasicsTransformerLM
from bohao.assignment2.cs336_basics.transformer_lm import TransformerLanguageModel


def benchmark() -> None:
    """
    Performs end-to-end benchmarking of the model's forward and backward passes.
    """
    # Define model size configurations based on the provided table
    model_configs = {
        "small": {
            "d_model": 768,
            "d_ff": 3072,
            "num_layers": 12,
            "num_heads": 12,
        },
        "medium": {
            "d_model": 1024,
            "d_ff": 4096,
            "num_layers": 24,
            "num_heads": 16,
        },
        "large": {
            "d_model": 1280,
            "d_ff": 5120,
            "num_layers": 36,
            "num_heads": 20,
        },
        "xl": {
            "d_model": 1600,
            "d_ff": 6400,
            "num_layers": 48,
            "num_heads": 25,
        },
        "2.7B": {
            "d_model": 2560,
            "d_ff": 10240,
            "num_layers": 32,
            "num_heads": 32,
        },
    }

    parser = argparse.ArgumentParser(
        description="Transformer Language Model Benchmarking."
    )
    parser.add_argument(
        "--warmup", type=int, default=5, help="Number of warm-up steps."
    )
    parser.add_argument(
        "--steps", type=int, default=10, help="Number of steps to time."
    )
    parser.add_argument(
        "--backward", action="store_true", help="Include backward pass in timing."
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="small",
        choices=model_configs.keys(),
        help="Model size to benchmark.",
    )
    parser.add_argument(
        "--bf16", action="store_true", help="Use mixed precision with BF16."
    )
    parser.add_argument(
        "--profile-memory",
        action="store_true",
        help="Profile memory usage and save a pickle file.",
    )
    args = parser.parse_args()

    config = model_configs[args.model_size]
    batch_size = 4
    vocab_size = 10000
    context_length = 512
    rope_theta = 10000.0

    # Check for CUDA availability
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available. Using GPU for benchmarking.")
        # Check for BF16 support
        if args.bf16 and torch.cuda.is_bf16_supported():
            print("BF16 is supported on this GPU. Enabling mixed precision.")
        elif args.bf16:
            print("Warning: BF16 is not supported on this GPU. Falling back to FP32.")
            args.bf16 = False
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU for benchmarking.")
        if args.profile_memory:
            print(
                "Warning: Memory profiling is only supported on CUDA-enabled devices. Disabling."
            )
            args.profile_memory = False

    try:
        # Instantiate the model and move to device
        model = TransformerLanguageModel(
            vocab_size=vocab_size,
            context_length=context_length,
            d_model=config["d_model"],
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            d_ff=config["d_ff"],
            rope_theta=rope_theta,
        )
        model.to(device)
        print(f"Model initialized ({args.model_size}) and moved to {device}.")

        # Create a random batch of data and move to device
        input_data = torch.randint(0, vocab_size, (batch_size, context_length)).to(
            device
        )
        target_data = torch.randint(0, vocab_size, (batch_size, context_length)).to(
            device
        )

        # Define loss function and optimizer
        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters())

        # Initialize GradScaler for mixed precision training
        if args.bf16:
            scaler = torch.cuda.amp.GradScaler()

        # --- Warm-up steps ---
        print(f"Starting {args.warmup} warm-up steps...")
        for _ in range(args.warmup):
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=args.bf16):
                output = model(input_data)
                if args.backward:
                    loss = loss_function(
                        output.view(-1, vocab_size), target_data.view(-1)
                    )

            if args.backward:
                if args.bf16:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            if device.type == "cuda":
                torch.cuda.synchronize()

        print("Warm-up complete. Starting timing.")

        # --- Start Memory Profiling ---
        if args.profile_memory:
            print("Starting memory history recording.")
            torch.cuda.memory._record_memory_history(max_entries=1000000)

        # --- Timing the execution steps ---
        step_times = []
        for _ in range(args.steps):
            optimizer.zero_grad()

            start_time = timeit.default_timer()
            with torch.cuda.amp.autocast(enabled=args.bf16):
                output = model(input_data)
                if args.backward:
                    loss = loss_function(
                        output.view(-1, vocab_size), target_data.view(-1)
                    )

            if args.backward:
                if args.bf16:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            if device.type == "cuda":
                torch.cuda.synchronize()
            end_time = timeit.default_timer()
            step_times.append(end_time - start_time)

        # --- Report the results ---
        avg_time = np.mean(step_times)
        std_dev = np.std(step_times)

        print("\n--- Benchmark Results ---")
        print(f"Model Size: {args.model_size.capitalize()}")
        print(
            f"Pass Type: {'Forward and Backward' if args.backward else 'Forward Only'}"
        )
        print(f"Precision: {'BF16' if args.bf16 else 'FP32'}")
        print(f"Total Steps Measured: {args.steps}")
        print(f"Average Time per Step: {avg_time:.6f} seconds")
        print(f"Standard Deviation: {std_dev:.6f} seconds")

        # --- Stop Memory Profiling and Save Snapshot ---
        if args.profile_memory:
            snapshot_filename = "memory_snapshot.pickle"
            torch.cuda.memory._dump_snapshot(snapshot_filename)
            torch.cuda.memory._record_memory_history(enabled=None)
            print(f"Memory snapshot saved to {snapshot_filename}")
            print("Memory history recording stopped.")

    except Exception as e:
        print(f"An error occurred during benchmarking: {e}")
        print("Full stack trace:")
        print(traceback.format_exc())


if __name__ == "__main__":
    benchmark()
