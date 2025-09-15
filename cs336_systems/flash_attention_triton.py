import torch
import triton
import triton.language as tl


# Triton kernel stub
@triton.jit
def flash_fwd_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    L_ptr,
    stride_qb,
    stride_qq,
    stride_qd,
    stride_kb,
    stride_kk,
    stride_kd,
    stride_vb,
    stride_vk,
    stride_vd,
    stride_ob,
    stride_oq,
    stride_od,
    stride_lb,
    stride_lq,
    N_QUERIES,
    N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(0, 1),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(0, 1),
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES, 1),
        strides=(stride_lq, 1),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, 1),
        order=(1, 0),
    )

    q_mask = (query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE) < N_QUERIES)[
        :, None
    ]
    q = tl.load(Q_block_ptr, mask=q_mask, other=0.0)

    o_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    m_i = tl.full((Q_TILE_SIZE,), -float("inf"), dtype=tl.float32)

    for _ in range(0, N_KEYS, K_TILE_SIZE):
        k = tl.load(K_block_ptr, mask=None, other=0.0)
        v = tl.load(V_block_ptr, mask=None, other=0.0)

        s = tl.dot(q, k) / scale

        m_i_new = tl.maximum(m_i, tl.max(s, 1))
        s = tl.exp(s - m_i_new[:, None])

        l_i = tl.exp(m_i - m_i_new) * l_i + tl.sum(s, 1)
        o_i = tl.exp(m_i - m_i_new)[:, None] * o_i + tl.dot(s, v)
        m_i = m_i_new

    o_final = o_i / tl.exp(l_i)[:, None]
    l_final = tl.log(l_i) + m_i

    row_mask = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE) < N_QUERIES

    tl.store(O_block_ptr, o_final, mask=row_mask[:, None])
    tl.store(L_block_ptr, l_final, mask=row_mask)


# torch.autograd.Function subclass stub
class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, scale, Q_TILE_SIZE=128, K_TILE_SIZE=128):
        # Store for backward pass (if needed)
        # ctx.save_for_backward(...)

        # Get tensor properties
        batch_size, N_QUERIES, D = Q.shape
        _, N_KEYS, _ = K.shape

        # Allocate output tensors
        O = torch.empty_like(Q)
        L = torch.empty((batch_size, N_QUERIES), device=Q.device, dtype=torch.float32)

        # Set up the launch grid
        grid = (triton.cdiv(N_QUERIES, Q_TILE_SIZE), batch_size)

        # Launch the Triton kernel
        flash_fwd_kernel[grid](
            Q,
            K,
            V,
            O,
            L,
            Q.stride(0),
            Q.stride(1),
            Q.stride(2),
            K.stride(0),
            K.stride(1),
            K.stride(2),
            V.stride(0),
            V.stride(1),
            V.stride(2),
            O.stride(0),
            O.stride(1),
            O.stride(2),
            L.stride(0),
            L.stride(1),
            N_QUERIES,
            N_KEYS,
            scale,
            D,
            Q_TILE_SIZE,
            K_TILE_SIZE,
        )
        return O, L

    @staticmethod
    def backward(ctx, grad_output, grad_L):
        # Backward pass logic goes here
        return None, None, None, None, None, None

# Adapter function to match the test suite's import structure
def get_flashattention_autograd_function_triton():
    return FlashAttentionTriton


def _attention_and_lse(q, k, v, is_causal=False):
    n_queries = q.shape[-2]
    n_keys = k.shape[-2]
    d = q.shape[-1]
    scale = 1 / (d ** 0.5)
    S = torch.einsum(q, k, '... q d, ... k d -> ... q k') * scale
    if is_causal:
        S = torch.where(
            torch.arange(n_queries, device=S.device)[None, :, None] >= torch.arange(n_keys, device=S.device)[None, None, :],
            S,
            -1e6
        )
    P = torch.softmax(S, dim=-1)
    o = torch.einsum(P, v, '... q k, ... k d -> ... q d')
    L = torch.logsumexp(S, dim=-1)
    return o, L


def _make_attn_inputs(device=None):
    torch.random.manual_seed(0)
    batch_size = 4
    n_queries = 128
    n_keys = 128
    D = 64
    q = torch.randn(batch_size, n_queries, D, device=device, requires_grad=True)
    k = torch.randn(batch_size, n_keys, D, device=device, requires_grad=True)
    v = torch.randn(batch_size, n_keys, D, device=device, requires_grad=True)
    _do = torch.randn(batch_size, n_queries, D, device=device)
    return q, k, v, _do


def _test_flash_forward_pass(impl, device="cpu", is_causal=False):
    q, k, v, _do = _make_attn_inputs(device)
    o, l = impl(q, k, v, is_causal)

    # In the original test, `l` is extracted from saved tensors.
    # The provided FlashAttentionTriton returns `o, l` directly.
    # We will use the direct return for simplicity.

    o_ref, l_ref = _attention_and_lse(q, k, v, is_causal)

    # Check for close approximation
    try:
        torch.testing.assert_close(o, o_ref, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(l, l_ref, rtol=1e-2, atol=1e-2)
        return True, None
    except AssertionError as e:
        return False, str(e)


def test_flash_forward_pass_triton(is_causal):
    # This is the actual test function from the user's prompt
    return _test_flash_forward_pass(
        get_flashattention_autograd_function_triton().apply,
        device="cuda",
        is_causal=is_causal,
    )


if __name__ == "__main__":
    import sys
    if not torch.cuda.is_available():
        print("A GPU must be available to run this test. Skipping.")
        sys.exit(0)

    print("Running FlashAttention Triton forward pass test...")
    print("--------------------------------------------------")

    # Test for non-causal attention
    is_causal = False
    print(f"Testing with is_causal={is_causal}...")
    success, error_msg = test_flash_forward_pass_triton(is_causal)
    if success:
        print("✅ Non-causal forward pass test PASSED!")
    else:
        print("❌ Non-causal forward pass test FAILED.")
        print("Error details:", error_msg)

    print("\n" + "=" * 50 + "\n")

    # Test for causal attention
    is_causal = True
    print(f"Testing with is_causal={is_causal}...")
    success, error_msg = test_flash_forward_pass_triton(is_causal)
    if success:
        print("✅ Causal forward pass test PASSED!")
    else:
        print("❌ Causal forward pass test FAILED.")
        print("Error details:", error_msg)
        print("\nNote: The provided Triton kernel does not implement causality, so this test will fail.")
