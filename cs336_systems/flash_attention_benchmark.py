import time
from itertools import product

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# Import your FlashAttention implementation
from bohao.assignment2.flash_attention_triton import FlashAttentionTriton


def pytorch_attention(Q, K, V, is_causal=True):
    """Standard PyTorch attention implementation with causal masking."""
    batch_size, seq_len, d_model = Q.shape
    scale = 1.0 / (d_model**0.5)

    # Compute attention scores
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

    # Apply causal mask
    if is_causal:
        mask = torch.tril(
            torch.ones(seq_len, seq_len, device=Q.device, dtype=torch.bool)
        )
        scores = scores.masked_fill(~mask, float("-inf"))

    # Apply softmax
    attn_weights = F.softmax(scores, dim=-1)

    # Apply attention to values
    out = torch.matmul(attn_weights, V)

    return out


def get_tile_sizes(seq_len, d_model):
    if d_model >= 128:
        # Use smaller tiles for large head dims
        q_tile = min(16, seq_len)
        k_tile = min(128, seq_len)
    elif d_model >= 64:
        q_tile = min(16, seq_len)
        k_tile = min(16, seq_len)
    else:
        # Can use larger tiles for small head dims
        q_tile = min(128, seq_len)
        k_tile = min(128, seq_len)

    # Ensure tile sizes are powers of 2 and don't exceed sequence length
    q_tile = min(q_tile, seq_len)
    k_tile = min(k_tile, seq_len)

    return q_tile, k_tile


def generate_inputs(batch_size, seq_len, d_model, dtype, device="cuda"):
    """Generate random inputs for benchmarking."""
    torch.manual_seed(42)  # For reproducibility

    Q = torch.randn(
        batch_size, seq_len, d_model, dtype=dtype, device=device, requires_grad=True
    )
    K = torch.randn(
        batch_size, seq_len, d_model, dtype=dtype, device=device, requires_grad=True
    )
    V = torch.randn(
        batch_size, seq_len, d_model, dtype=dtype, device=device, requires_grad=True
    )

    return Q, K, V


def benchmark_forward(fn, *args, **kwargs):
    """Benchmark forward pass using triton.testing.do_bench."""

    def wrapper():
        return fn(*args, **kwargs)

    # Warmup
    for _ in range(10):
        wrapper()

    torch.cuda.synchronize()
    return triton.testing.do_bench(wrapper, warmup=25, rep=100)


def benchmark_backward(fn, grad_output):
    """Benchmark backward pass."""

    def wrapper():
        output = fn()
        if isinstance(output, tuple):
            output = output[0]  # For Triton implementation
        output.backward(grad_output)
        return output

    # Warmup
    for _ in range(10):
        try:
            wrapper()
        except:
            pass

    torch.cuda.synchronize()
    return triton.testing.do_bench(wrapper, warmup=25, rep=100)


def benchmark_end_to_end(fn, grad_output):
    """Benchmark end-to-end forward + backward pass."""

    def wrapper():
        output = fn()
        if isinstance(output, tuple):
            output = output[0]  # For Triton implementation
        output.backward(grad_output)
        return output

    # Warmup
    for _ in range(10):
        wrapper()

    torch.cuda.synchronize()
    return triton.testing.do_bench(wrapper, warmup=25, rep=100)


def run_benchmark():
    """Run the complete benchmark suite."""
    # Configuration
    batch_size = 1
    seq_lengths = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    d_models = [16, 32, 64, 128]
    dtypes = [torch.bfloat16, torch.float32]
    device = "cuda"

    results = []

    print("Starting FlashAttention vs PyTorch Benchmark")
    print("=" * 60)

    total_configs = len(seq_lengths) * len(d_models) * len(dtypes)
    config_count = 0

    for seq_len, d_model, dtype in product(seq_lengths, d_models, dtypes):
        config_count += 1
        print(
            f"\nConfig {config_count}/{total_configs}: seq_len={seq_len}, d_model={d_model}, dtype={dtype}"
        )

        try:
            # Generate inputs
            Q, K, V = generate_inputs(batch_size, seq_len, d_model, dtype, device)
            scale = 1.0 / (d_model**0.5)

            # Get appropriate tile sizes
            q_tile, k_tile = get_tile_sizes(seq_len, d_model)

            # Create gradient output for backward pass
            grad_output = torch.randn_like(Q)

            # Define forward functions
            def triton_forward():
                return FlashAttentionTriton.apply(Q, K, V, scale, True, q_tile, k_tile)

            def pytorch_forward():
                return pytorch_attention(Q, K, V, is_causal=True)

            # Benchmark forward pass
            print("  Benchmarking forward passes...")
            triton_fwd_time = benchmark_forward(triton_forward)
            pytorch_fwd_time = benchmark_forward(pytorch_forward)

            # Reset gradients
            for tensor in [Q, K, V]:
                if tensor.grad is not None:
                    tensor.grad.zero_()

            # Benchmark backward pass
            print("  Benchmarking backward passes...")
            triton_bwd_time = benchmark_backward(triton_forward, grad_output)

            # Reset gradients for PyTorch
            for tensor in [Q, K, V]:
                if tensor.grad is not None:
                    tensor.grad.zero_()

            pytorch_bwd_time = benchmark_backward(pytorch_forward, grad_output)

            # Reset gradients
            for tensor in [Q, K, V]:
                if tensor.grad is not None:
                    tensor.grad.zero_()

            # Benchmark end-to-end
            print("  Benchmarking end-to-end passes...")
            triton_e2e_time = benchmark_end_to_end(triton_forward, grad_output)

            # Reset gradients for PyTorch
            for tensor in [Q, K, V]:
                if tensor.grad is not None:
                    tensor.grad.zero_()

            pytorch_e2e_time = benchmark_end_to_end(pytorch_forward, grad_output)

            # Store results
            result = {
                "seq_len": seq_len,
                "d_model": d_model,
                "dtype": str(dtype).split(".")[-1],
                "q_tile": q_tile,
                "k_tile": k_tile,
                "triton_fwd_ms": triton_fwd_time,
                "pytorch_fwd_ms": pytorch_fwd_time,
                "triton_bwd_ms": triton_bwd_time,
                "pytorch_bwd_ms": pytorch_bwd_time,
                "triton_e2e_ms": triton_e2e_time,
                "pytorch_e2e_ms": pytorch_e2e_time,
                "fwd_speedup": pytorch_fwd_time / triton_fwd_time,
                "bwd_speedup": pytorch_bwd_time / triton_bwd_time,
                "e2e_speedup": pytorch_e2e_time / triton_e2e_time,
            }

            results.append(result)

            print(
                f"  Forward: Triton {triton_fwd_time:.3f}ms vs PyTorch {pytorch_fwd_time:.3f}ms (speedup: {result['fwd_speedup']:.2f}x)"
            )
            print(
                f"  Backward: Triton {triton_bwd_time:.3f}ms vs PyTorch {pytorch_bwd_time:.3f}ms (speedup: {result['bwd_speedup']:.2f}x)"
            )
            print(
                f"  End-to-end: Triton {triton_e2e_time:.3f}ms vs PyTorch {pytorch_e2e_time:.3f}ms (speedup: {result['e2e_speedup']:.2f}x)"
            )

        except Exception as e:
            print(f"  Skipping due to error: {str(e)}")
            continue

    # Create DataFrame and save results
    df = pd.DataFrame(results)

    if len(df) > 0:
        # Display summary table
        print("\n" + "=" * 100)
        print("BENCHMARK RESULTS SUMMARY")
        print("=" * 100)

        # Format and display the main results table
        display_df = df.copy()

        # Round numerical columns for better display
        numerical_cols = [
            "triton_fwd_ms",
            "pytorch_fwd_ms",
            "triton_bwd_ms",
            "pytorch_bwd_ms",
            "triton_e2e_ms",
            "pytorch_e2e_ms",
            "fwd_speedup",
            "bwd_speedup",
            "e2e_speedup",
        ]

        for col in numerical_cols:
            if col.endswith("_speedup"):
                display_df[col] = display_df[col].round(2)
            else:
                display_df[col] = display_df[col].round(3)

        # Print the complete table
        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", None)
        print(display_df.to_string(index=False))

        # Save to CSV
        df.to_csv("flashattention_benchmark_results.csv", index=False)
        print(f"\nResults saved to 'flashattention_benchmark_results.csv'")

        # Print summary statistics
        print("\n" + "=" * 50)
        print("SUMMARY STATISTICS")
        print("=" * 50)
        print(f"Average Forward Speedup: {df['fwd_speedup'].mean():.2f}x")
        print(f"Average Backward Speedup: {df['bwd_speedup'].mean():.2f}x")
        print(f"Average End-to-End Speedup: {df['e2e_speedup'].mean():.2f}x")
        print(f"Max Forward Speedup: {df['fwd_speedup'].max():.2f}x")
        print(f"Max Backward Speedup: {df['bwd_speedup'].max():.2f}x")
        print(f"Max End-to-End Speedup: {df['e2e_speedup'].max():.2f}x")

    else:
        print("No successful benchmark results obtained.")


if __name__ == "__main__":
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. This benchmark requires a GPU.")
        exit(1)

    # Check if we're on an H100 (optional check)
    gpu_name = torch.cuda.get_device_name()
    print(f"Running on GPU: {gpu_name}")

    # Run the benchmark
    run_benchmark()
