# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-strict

import argparse
import time
import traceback

import numpy as np

import torch


def self_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Performs a single-head self-attention operation.

    Args:
        q (torch.Tensor): Query tensor of shape (B, S, D).
        k (torch.Tensor): Key tensor of shape (B, S, D).
        v (torch.Tensor): Value tensor of shape (B, S, D).

    Returns:
        torch.Tensor: The output tensor of the attention operation.
    """
    # Get dimensions
    batch_size, seq_len, d_model = q.shape

    # Calculate attention scores
    # (B, S, D) x (B, D, S) -> (B, S, S)
    scores = torch.bmm(q, k.transpose(1, 2))

    # Scale the scores
    scale_factor = d_model**-0.5
    scores = scores * scale_factor

    # Apply softmax to get attention weights
    attention_weights = torch.softmax(scores, dim=-1)

    # Multiply with values
    # (B, S, S) x (B, S, D) -> (B, S, D)
    output = torch.bmm(attention_weights, v)

    return output


def benchmark() -> None:
    """
    Performs end-to-end benchmarking of the self-attention's forward and backward passes.
    """
    parser = argparse.ArgumentParser(description="Self-Attention Benchmarking.")
    parser.add_argument(
        "--warmup", type=int, default=5, help="Number of warm-up steps."
    )
    parser.add_argument(
        "--steps", type=int, default=100, help="Number of steps to time."
    )
    args = parser.parse_args()

    # Define the dimensions to iterate through
    d_models = [16, 32, 64, 128]
    sequence_lengths = [256, 1024, 4096, 8192, 16384]
    batch_size = 8

    # Check for CUDA availability
    if not torch.cuda.is_available():
        print(
            "CUDA is not available. This script requires a GPU for meaningful results."
        )
        return

    device = torch.device("cuda")
    print("CUDA is available. Using GPU for benchmarking.")

    # Loop through the Cartesian product of dimensions
    for d_model in d_models:
        for seq_len in sequence_lengths:
            print("-" * 50)
            print(f"Benchmarking: d_model={d_model}, sequence_length={seq_len}")

            try:
                # Create random inputs and move to device
                q_data = torch.randn(
                    batch_size, seq_len, d_model, device=device, requires_grad=True
                )
                k_data = torch.randn(batch_size, seq_len, d_model, device=device)
                v_data = torch.randn(batch_size, seq_len, d_model, device=device)

                # --- Warm-up steps ---
                for _ in range(args.warmup):
                    output = self_attention(q_data, k_data, v_data)
                    output.sum().backward()
                    torch.cuda.synchronize()

                print("Warm-up complete. Starting timing.")

                # --- Timing the forward pass ---
                forward_times = []
                for _ in range(args.steps):
                    start_time = time.time()
                    output = self_attention(q_data, k_data, v_data)
                    torch.cuda.synchronize()
                    end_time = time.time()
                    forward_times.append(end_time - start_time)

                avg_forward = np.mean(forward_times)
                std_forward = np.std(forward_times)
                print(f"Average Forward Time: {avg_forward * 1000:.4f} ms")
                print(f"Standard Deviation: {std_forward * 1000:.4f} ms")

                # --- Memory and backward pass ---
                torch.cuda.empty_cache()
                q_data = torch.randn(
                    batch_size, seq_len, d_model, device=device, requires_grad=True
                )
                k_data = torch.randn(batch_size, seq_len, d_model, device=device)
                v_data = torch.randn(batch_size, seq_len, d_model, device=device)

                output = self_attention(q_data, k_data, v_data)

                # Measure memory usage before backward pass
                memory_allocated_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                print(
                    f"Memory allocated before backward pass: {memory_allocated_mb:.2f} MB"
                )

                # Time the backward pass
                backward_times = []
                for _ in range(args.steps):
                    start_time = time.time()
                    output.sum().backward(
                        retain_graph=True
                    )  # retain_graph for multiple backward passes
                    torch.cuda.synchronize()
                    end_time = time.time()
                    backward_times.append(end_time - start_time)

                avg_backward = np.mean(backward_times)
                std_backward = np.std(backward_times)
                print(f"Average Backward Time: {avg_backward * 1000:.4f} ms")
                print(f"Standard Deviation: {std_backward * 1000:.4f} ms")

            except torch.cuda.OutOfMemoryError as e:
                print(f"Out of memory for this configuration.")
                print("Full stack trace:")
                print(traceback.format_exc())

            except Exception as e:
                print(f"An error occurred: {e}")
                print("Full stack trace:")
                print(traceback.format_exc())
    print("-" * 50)


if __name__ == "__main__":
    benchmark()
