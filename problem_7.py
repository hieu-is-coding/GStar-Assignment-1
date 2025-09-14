import torch
import triton
import triton.language as tl
import math

@triton.jit
def process_kv_block(
    start_n, acc, m_i, l_i,
    q_block, q_offsets, qk_scale,
    K_ptr, V_ptr,
    batch_idx, kv_head_idx,
    k_stride_b, k_stride_h, k_stride_s,
    v_stride_b, v_stride_h, v_stride_s,
    SEQ_LEN, 
    HEAD_DIM, 
    BLOCK_M: tl.constexpr, 
    BLOCK_N: tl.constexpr, 
    WINDOW_SIZE: tl.constexpr, 
    SINK_SIZE: tl.constexpr,
    process_name,
):
    """A kernel function for less code lines"""
    k_offsets = start_n + tl.arange(0, BLOCK_N)

    k_valid = (k_offsets[None, :] >=0) & (k_offsets[None, :] < SEQ_LEN)
    k_ptrs = K_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h + k_offsets[None, :] * k_stride_s + tl.arange(0, HEAD_DIM)[:, None]
    k_block = tl.load(k_ptrs, mask=k_valid, other=0.0).to(tl.float32)
    # tl.static_print(f"{process_name} - k_offsets, k_valid, k_ptrs, k_block: ", k_offsets, k_valid, k_ptrs, k_block)

    v_valid = (k_offsets[:, None] >= 0) & (k_offsets[:, None] < SEQ_LEN)
    v_ptrs = V_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h + k_offsets[:, None] * v_stride_s + tl.arange(0, HEAD_DIM)[None, :]
    v_block = tl.load(v_ptrs, mask=v_valid, other=0.0).to(tl.float32)
    # tl.static_print(f"{process_name} - v_valid, v_ptrs, v_block: ", v_valid, v_ptrs, v_block)

    s_ij = tl.dot(q_block, k_block)
    s_ij *= qk_scale
    # tl.static_print(f"{process_name} - s_ij: ", s_ij)
    # tl.static_print("s_ij shape:", s_ij.shape)

    # Apply comnined masking
    causal_mask = q_offsets[:, None] >= k_offsets[None, :]
    swa_mask = (q_offsets[:, None] - k_offsets[None, :]) <= (WINDOW_SIZE - 1)
    sink_mask = k_offsets[None, :] < SINK_SIZE
    combined_mask = k_valid & causal_mask & (swa_mask | sink_mask)
    # combined_mask = k_valid & causal_mask & swa_mask & sink_mask
    # tl.static_print(f"{process_name} - s_ij: ", combined_mask)
    s_ij = tl.where(combined_mask, s_ij, -float('inf'))

    # Reuse online softmax with small change in m_new update
    row_max = tl.maximum(m_i, tl.max(s_ij, axis=1))
    row_oke = row_max > -float('inf')
    m_new = tl.maximum(row_oke, row_max)
    exp_diff = tl.where(row_oke, tl.exp2(m_i - m_new), 1.0)
    # tl.static_print(f"{process_name} - row_max, row_oke, m_new: ", row_max, row_oke, m_new)
    acc = acc * exp_diff[:, None]
    l_i = l_i * exp_diff
    # tl.static_print(f"{process_name} - acc, l_i: ", acc, l_i)
    p_ij = tl.where(row_oke[:, None], tl.exp2(s_ij - m_new[:, None]), 0.0)
    # tl.static_print(f"{process_name} - p_ij: ", p_ij)
    acc += tl.dot(p_ij, v_block)
    l_i += tl.sum(p_ij, axis=1)
    # tl.static_print(f"{process_name} - acc, l_i: ", acc, l_i)
    m_i = m_new
    return acc, m_i, l_i

@triton.jit
def _flash_attention_forward_swa_kernel(
    # Pointers to Tensors
    Q_ptr, K_ptr, V_ptr, O_ptr,
    # Stride information for tensors
    q_stride_b, q_stride_h, q_stride_s,
    k_stride_b, k_stride_h, k_stride_s,
    v_stride_b, v_stride_h, v_stride_s,
    # Kernel parameters
    softmax_scale,
    SEQ_LEN,
    N_Q_HEADS,
    N_KV_HEADS,
    WINDOW_SIZE: tl.constexpr,
    SINK_SIZE: tl.constexpr,
    # Constexpr tile sizes
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Triton kernel for the forward pass of causal FlashAttention with GQA, Sliding Window Attention, and Attention Sink.
    """
    # 1. Identify the block of queries and the batch/head to be processed.
    q_block_idx = tl.program_id(axis=0)
    batch_head_idx = tl.program_id(axis=1)
    
    batch_idx = batch_head_idx // N_Q_HEADS
    q_head_idx = batch_head_idx % N_Q_HEADS

    # --- GQA Logic: Map Query Head to Shared K/V Head ---
    num_groups = N_Q_HEADS // N_KV_HEADS
    kv_head_idx = q_head_idx // num_groups

    # 2. Initialize accumulators in SRAM.
    m_i = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # 3. Load the block of queries (Q_i).
    q_offsets = (q_block_idx * BLOCK_M + tl.arange(0, BLOCK_M))
    q_ptrs = Q_ptr + batch_idx * q_stride_b + q_head_idx * q_stride_h + \
             (q_offsets[:, None] * q_stride_s + tl.arange(0, HEAD_DIM)[None, :])
    q_block = tl.load(q_ptrs, mask=q_offsets[:, None] < SEQ_LEN, other=0.0)
    
    qk_scale = softmax_scale * 1.44269504

    # --- STUDENT IMPLEMENTATION REQUIRED HERE ---
    # Combine the GQA, SWA, and Sink logic.
    # Combine all code from previous problems, and add the sink logic.
    # You should have 3 phases:
    # Set the correct start bound
    window_start = tl.maximum(0, q_block_idx * BLOCK_M - (WINDOW_SIZE - 1))
    window_start = (window_start // BLOCK_N) * BLOCK_N
    # Set q_block to float32 for type matching
    q_block = q_block.to(tl.float32)
    
    # 1. Phase 0: Sink blocks that are before the sliding window
    sink_end = tl.minimum(tl.minimum(SINK_SIZE, SEQ_LEN), window_start)
    if sink_end > 0:
        for start_n in range(0, sink_end, BLOCK_N):
            acc, m_i, l_i = process_kv_block(
                start_n, acc, m_i, l_i, 
                q_block, q_offsets, qk_scale,
                K_ptr, V_ptr, batch_idx, kv_head_idx,
                k_stride_b, k_stride_h, k_stride_s,
                v_stride_b, v_stride_h, v_stride_s,
                SEQ_LEN, HEAD_DIM, BLOCK_M, BLOCK_N, WINDOW_SIZE, SINK_SIZE,
                "Sink process"
            )
        # tl.static_print(acc, m_i, l_i)
    # 2. Phase 1: Off-Diagonal Blocks (within the window)
    for start_n in range(window_start, q_block_idx * BLOCK_M, BLOCK_N):
        acc, m_i, l_i = process_kv_block(
            start_n, acc, m_i, l_i, 
            q_block, q_offsets, qk_scale,
            K_ptr, V_ptr, batch_idx, kv_head_idx,
            k_stride_b, k_stride_h, k_stride_s,
            v_stride_b, v_stride_h, v_stride_s,
            SEQ_LEN, HEAD_DIM, BLOCK_M, BLOCK_N, WINDOW_SIZE, SINK_SIZE,
            "Off-Diagonal Blocks"
        )
    # 3. Phase 2: Diagonal Blocks
    diag_start = q_block_idx * BLOCK_M
    for start_n in range(diag_start, tl.minimum((q_block_idx + 1) * BLOCK_M, SEQ_LEN), BLOCK_N):
        acc, m_i, l_i = process_kv_block(
            start_n, acc, m_i, l_i, 
            q_block, q_offsets, qk_scale,
            K_ptr, V_ptr, batch_idx, kv_head_idx,
            k_stride_b, k_stride_h, k_stride_s,
            v_stride_b, v_stride_h, v_stride_s,
            SEQ_LEN, HEAD_DIM, BLOCK_M, BLOCK_N, WINDOW_SIZE, SINK_SIZE,
            "Diagonal Blocks"
        )
    # --- END OF STUDENT IMPLEMENTATION ---

    # 4. Normalize and write the final output block.
    l_i_safe = tl.where(l_i == 0, 1.0, l_i)
    acc = acc / l_i_safe[:, None]
    
    o_ptrs = O_ptr + batch_idx * q_stride_b + q_head_idx * q_stride_h + \
             (q_offsets[:, None] * q_stride_s + tl.arange(0, HEAD_DIM)[None, :])
             
    tl.store(o_ptrs, acc.to(O_ptr.dtype.element_ty), mask=q_offsets[:, None] < SEQ_LEN)


def flash_attention_forward(q, k, v, is_causal=True, window_size=128, sink_size=4):
    """
    Python wrapper for the SWA-enabled GQA causal FlashAttention kernel with attention sink support.
    """
    # Shape checks
    batch, n_q_heads, seq_len, head_dim = q.shape
    _, n_kv_heads, _, _ = k.shape
    
    # Assertions
    assert q.shape[0] == v.shape[0] and q.shape[2] == v.shape[2] and q.shape[3] == v.shape[3]
    assert k.shape == v.shape
    assert head_dim <= 128
    assert n_q_heads % n_kv_heads == 0
    assert is_causal, "This kernel only supports causal attention"
    
    o = torch.empty_like(q)
    softmax_scale = 1.0 / math.sqrt(head_dim)
    
    BLOCK_M, BLOCK_N = 128, 64
    grid = (triton.cdiv(seq_len, BLOCK_M), batch * n_q_heads)

    _flash_attention_forward_swa_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        softmax_scale,
        seq_len,
        n_q_heads,
        n_kv_heads,
        WINDOW_SIZE=window_size,
        SINK_SIZE=sink_size,
        HEAD_DIM=head_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    return o