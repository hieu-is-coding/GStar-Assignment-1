import torch
import triton
import triton.language as tl
import math
from typing import Optional

@triton.jit
def _flash_attention_forward_swa_kernel(
    # Pointers to Tensors
    Q_ptr, K_ptr, V_ptr, O_ptr, M_ptr,
    # Stride information for tensors
    q_stride_b, q_stride_h, q_stride_s,
    k_stride_b, k_stride_h, k_stride_s,
    v_stride_b, v_stride_h, v_stride_s,
    o_stride_b, o_stride_h, o_stride_s,
    m_stride_b, m_stride_h, m_stride_s,
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
    # Phase 0: Set up same as problem_8
    # Identify the block of queries and the batch/head to be processed.
    q_block_idx = tl.program_id(0)
    batch_head_idx = tl.program_id(1)

    batch_idx = batch_head_idx // N_Q_HEADS
    q_head_idx = batch_head_idx % N_Q_HEADS

    heads_per_group = N_Q_HEADS // N_KV_HEADS
    kv_head_idx = q_head_idx // heads_per_group
    # tl.static_print("heads_per_group, kv_head_idx: ", heads_per_group, kv_head_idx)

    # Initialize accumulators in SRAM.
    m_i = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # Load the block of queries.
    q_offsets = q_block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    q_ptrs = Q_ptr + batch_idx * q_stride_b + q_head_idx * q_stride_h + q_offsets[:, None] * q_stride_s + tl.arange(0, HEAD_DIM)[None, :]
    q_mask = q_offsets < SEQ_LEN
    q_block = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0).to(tl.float32)
    # tl.static_print("q_maks, q_block: ", q_maks, q_block)

    # Phase 1: Off-diagonal blocks
    for start_n in range(0, tl.minimum(q_block_idx * BLOCK_M, SEQ_LEN), BLOCK_N):
        k_offsets = start_n + tl.arange(0, BLOCK_N)
        padding_mask = k_offsets < SEQ_LEN
        # tl.static_print("Off-Diagonal Blocks - k_offsets, padding_mask : ", k_offsets, padding_mask)

        # Masking application
        check_sink = k_offsets[None, :] < SINK_SIZE
        check_window = k_offsets[None, :] > (q_offsets[:, None] - WINDOW_SIZE)
        causal_mask = k_offsets[None, :] <= q_offsets[:, None]
        # tl.static_print("Off-Diagonal Blocks - check_sink, check_window, causal_mask: ", check_sink, check_window, causal_mask)
        
        combined_mask = causal_mask & (check_sink | check_window)
        combined_mask &= (q_mask[:, None] & padding_mask[None, :])
        # combined_mask |= (q_mask[:, None] & padding_mask[None, :])
        # tl.static_print("Off-Diagonal Blocks - combined_mask: ", combined_mask)

        # Load K and V blocks
        k_ptrs = K_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h + \
                 (k_offsets[None, :] * k_stride_s + tl.arange(0, HEAD_DIM)[:, None])
        v_ptrs = V_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h + \
                 (k_offsets[:, None] * v_stride_s + tl.arange(0, HEAD_DIM)[None, :])

        k_block = tl.load(k_ptrs, mask=padding_mask[None, :], other=0.0).to(tl.float32)
        v_block = tl.load(v_ptrs, mask=padding_mask[:, None], other=0.0).to(tl.float32)
        # tl.static_print("Off-Diagonal Blocks - k_block, v_block: ", k_block, v_block)

        # Online softmax
        s_ij = tl.dot(q_block, k_block) * softmax_scale
        s_ij = tl.where(combined_mask, s_ij, -float('inf'))
        # tl.static_print("Off-Diagonal Blocks - s_ij: ", s_ij)

        row_max = tl.max(s_ij, axis=1)
        row_oke = row_max > -float('inf')
        m_new = tl.where(row_oke, tl.maximum(m_i, row_max), m_i)
        # tl.static_print("Off-Diagonal Blocks - m_new: ", m_new)
        exp_diff = tl.where(row_oke, tl.exp(m_i - m_new), 1.0)
        acc *= exp_diff[:, None]
        l_i *= exp_diff
        # tl.static_print("Off-Diagonal Blocks - acc, l_i: ", acc, l_i)
        p_ij = tl.where(row_oke[:, None], tl.exp(s_ij - m_new[:, None]), 0.0)
        # tl.static_print("-Off-Diagonal Blocks - p_ij: ", p_ij)
        acc += tl.dot(p_ij, v_block)
        l_i += tl.sum(p_ij, axis=1)
        # tl.static_print("Off-Diagonal Blocks - acc, l_i: ", acc, l_i)
        m_i = m_new

    # Phase 2: Diagonal blocks
    diag_start = (q_block_idx * BLOCK_M) // BLOCK_N * BLOCK_N
    for start_n in range(diag_start, tl.minimum((q_block_idx + 1) * BLOCK_M, SEQ_LEN), BLOCK_N):
        k_offsets = start_n + tl.arange(0, BLOCK_N)
        padding_mask = k_offsets < SEQ_LEN
        # tl.static_print("Diagonal Blocks - k_offsets, padding_mask : ", k_offsets, padding_mask)

        # Masking application
        check_sink = k_offsets[None, :] < SINK_SIZE
        check_window = k_offsets[None, :] > (q_offsets[:, None] - WINDOW_SIZE)
        causal_mask = k_offsets[None, :] <= q_offsets[:, None]
        # tl.static_print("Diagonal Blocks - check_sink, check_window, causal_mask: ", check_sink, check_window, causal_mask)
        
        combined_mask = causal_mask & (check_sink | check_window)
        combined_mask &= (q_mask[:, None] & padding_mask[None, :])
        # combined_mask |= (q_mask[:, None] & padding_mask[None, :])
        # tl.static_print("Diagonal Blocks - combined_mask: ", combined_mask)

        # Load K and V blocks
        k_ptrs = K_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h + \
                 (k_offsets[None, :] * k_stride_s + tl.arange(0, HEAD_DIM)[:, None])
        v_ptrs = V_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h + \
                 (k_offsets[:, None] * v_stride_s + tl.arange(0, HEAD_DIM)[None, :])

        k_block = tl.load(k_ptrs, mask=padding_mask[None, :], other=0.0).to(tl.float32)
        v_block = tl.load(v_ptrs, mask=padding_mask[:, None], other=0.0).to(tl.float32)
        # tl.static_print("Diagonal Blocks - k_block, v_block: ", k_block, v_block)

        # Online softmax
        s_ij = tl.dot(q_block, k_block) * softmax_scale
        s_ij = tl.where(combined_mask, s_ij, -float('inf'))
        # tl.static_print("Diagonal Blocks - s_ij: ", s_ij)

        row_max = tl.max(s_ij, axis=1)
        row_oke = row_max > -float('inf')
        m_new = tl.where(row_oke, tl.maximum(m_i, row_max), m_i)
        # tl.static_print("Diagonal Blocks - m_new: ", m_new)
        exp_diff = tl.where(row_oke, tl.exp(m_i - m_new), 1.0)
        acc *= exp_diff[:, None]
        l_i *= exp_diff
        # tl.static_print("Diagonal Blocks - acc, l_i: ", acc, l_i)
        p_ij = tl.where(row_oke[:, None], tl.exp(s_ij - m_new[:, None]), 0.0)
        # tl.static_print("-Diagonal Blocks - p_ij: ", p_ij)
        acc += tl.dot(p_ij, v_block)
        l_i += tl.sum(p_ij, axis=1)
        # tl.static_print("Diagonal Blocks - acc, l_i: ", acc, l_i)
        m_i = m_new

    # Normalize and store output
    l_i_safe = tl.where(l_i == 0, 1.0, l_i)
    acc = acc / l_i_safe[:, None]
    
    o_ptrs = O_ptr + batch_idx * o_stride_b + q_head_idx * o_stride_h + q_offsets[:, None] * o_stride_s + tl.arange(0, HEAD_DIM)[None, :]
    tl.store(o_ptrs, acc.to(O_ptr.dtype.element_ty), mask=q_mask[:, None])

    # Store log-sum-exp for backward pass
    lse = m_i + tl.log(l_i_safe)
    m_ptrs = M_ptr + batch_idx * m_stride_b + q_head_idx * m_stride_h + q_offsets
    tl.store(m_ptrs, lse, mask=q_mask)

@triton.jit
def _flash_attention_backward_swa_kernel(
    # In/Out Pointers
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, M_ptr,
    dQ_ptr, dK_ptr, dV_ptr,
    # Strides
    q_stride_b, q_stride_h, q_stride_s,
    k_stride_b, k_stride_h, k_stride_s,
    v_stride_b, v_stride_h, v_stride_s,
    o_stride_b, o_stride_h, o_stride_s,
    do_stride_b, do_stride_h, do_stride_s,
    m_stride_b, m_stride_h, m_stride_s,
    dq_stride_b, dq_stride_h, dq_stride_s,
    dk_stride_b, dk_stride_h, dk_stride_s,
    dv_stride_b, dv_stride_h, dv_stride_s,
    # Parameters
    softmax_scale,
    BATCH_SIZE: int,
    N_Q_HEADS: int,
    N_KV_HEADS: int,
    SEQ_LEN: int,
    WINDOW_SIZE: tl.constexpr,
    SINK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    # Tile Sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Kinda same with problem 8 (except for masking)
    # Get program and block indices
    q_block_idx = tl.program_id(0)
    batch_head_idx = tl.program_id(1)

    # Decompose batch/head indices for GQA
    batch_idx = batch_head_idx // N_Q_HEADS
    q_head_idx = batch_head_idx % N_Q_HEADS
    heads_per_group = N_Q_HEADS // N_KV_HEADS
    kv_head_idx = q_head_idx // heads_per_group
    # tl.static_print("heads_per_group, kv_head_idx: ", heads_per_group, kv_head_idx)

    # Define offsets for the current query block
    q_offsets = q_block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    q_mask = q_offsets < SEQ_LEN

    # Load Q, dO, O, and LSE for the current query block
    q_ptrs = Q_ptr + batch_idx * q_stride_b + q_head_idx * q_stride_h + q_offsets[:, None] * q_stride_s + tl.arange(0, HEAD_DIM)[None, :]
    q_block = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0).to(tl.float32)
    # tl.static_print("Backward - q_block: ", q_block)

    do_ptrs = dO_ptr + batch_idx * do_stride_b + q_head_idx * do_stride_h + q_offsets[:, None] * do_stride_s + tl.arange(0, HEAD_DIM)[None, :]
    do_block = tl.load(do_ptrs, mask=q_mask[:, None], other=0.0).to(tl.float32)
    # tl.static_print("Backward - do_block: ", do_block)

    o_ptrs = O_ptr + batch_idx * o_stride_b + q_head_idx * o_stride_h + q_offsets[:, None] * o_stride_s + tl.arange(0, HEAD_DIM)[None, :]
    o_block = tl.load(o_ptrs, mask=q_mask[:, None], other=0.0).to(tl.float32)
    # tl.static_print("Backward - o_block: ", o_block) 

    m_ptrs = M_ptr + batch_idx * m_stride_b + q_head_idx * m_stride_h + q_offsets
    m_i = tl.load(m_ptrs, mask=q_mask, other=-float('inf'))
    # tl.static_print("Backward - m_i: ", m_i) 

    #   1. Precompute delta = sum(dO * O)
    delta_i = tl.sum(do_block * o_block, axis=1)

    # Initialize dQ accumulator
    dq_acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    #   2. Recompute attention probabilities P = softmax(QK^T)
    for start_n in range(0, q_block_idx * BLOCK_M, BLOCK_N):
        k_offsets = start_n + tl.arange(0, BLOCK_N)
        padding_mask = k_offsets < SEQ_LEN
        # tl.static_print("Backward - padding_mask: ", padding_mask)

        # Create the combined attention mask (Flash Attention + Sliding Attention + Attention Sinks)
        check_sink = k_offsets[None, :] < SINK_SIZE
        check_window = k_offsets[None, :]> (q_offsets[:, None] - WINDOW_SIZE)
        causal_mask = k_offsets[None, :] <= q_offsets[:, None]
        combined_mask = causal_mask & (check_sink | check_window)
        combined_mask &= padding_mask[None, :]
        # tl.static_print("Backward - combined_mask: ", combined_mask)

        # Load K and V blocks
        k_ptrs = K_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h + \
                 (k_offsets[None, :] * k_stride_s + tl.arange(0, HEAD_DIM)[:, None])
        v_ptrs = V_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h + \
                 (k_offsets[:, None] * v_stride_s + tl.arange(0, HEAD_DIM)[None, :])
        
        k_block = tl.load(k_ptrs, mask=padding_mask[None, :], other=0.0).to(tl.float32)
        v_block = tl.load(v_ptrs, mask=padding_mask[:, None], other=0.0).to(tl.float32)
        # tl.static_print("Backward - k_block, v_block: ", k_block, v_block)

        # Recompute attention scores (S) and probabilities (P)
        s_ij = tl.dot(q_block, k_block) * softmax_scale
        s_ij = tl.where(combined_mask, s_ij, -float('inf'))
        #  tl.static_print("Backward - s_ij: ", s_ij.shape)

        p_ij = tl.exp(s_ij - m_i[:, None])
        # tl.static_print("Backward - p_ij: ", p_ij)

        #   3. Use delta + dO to accumulate gradients for dq, dk, dv
        # Compute gradient of scores (dS)
        dS_ij_scaled = p_ij * (tl.dot(do_block, tl.trans(v_block)) - delta_i[:, None])

        # Accumulate dQ
        dq_acc += tl.dot(dS_ij_scaled, tl.trans(k_block)) * softmax_scale

        # Compute and atomically add dK
        dk_add = tl.dot(tl.trans(dS_ij_scaled), q_block) * softmax_scale
        dk_ptrs = dK_ptr + batch_idx * dk_stride_b + kv_head_idx * dk_stride_h + k_offsets[:, None] * dk_stride_s + tl.arange(0, HEAD_DIM)[None, :]
        # tl.static_print("Backward - dk_ptrs, dk_add: ", dk_ptrs, dk_add)
        tl.atomic_add(dk_ptrs, dk_add, mask=padding_mask[:, None])

        # Compute and atomically add dV
        dv_add = tl.dot(tl.trans(p_ij.to(do_block.dtype)), do_block)
        dv_ptrs = dV_ptr + batch_idx * dv_stride_b + kv_head_idx * dv_stride_h + k_offsets[:, None] * v_stride_s + tl.arange(0, HEAD_DIM)[None, :]
        # tl.static_print("Backward - dv_ptrs, dv_add: ", dv_ptrs, dv_add)
        tl.atomic_add(dv_ptrs, dv_add, mask=padding_mask[:, None])

    diag_start = (q_block_idx * BLOCK_M) // BLOCK_N * BLOCK_N
    for start_n in range(diag_start, tl.minimum((q_block_idx + 1) * BLOCK_M, SEQ_LEN), BLOCK_N):
        k_offsets = start_n + tl.arange(0, BLOCK_N)
        padding_mask = k_offsets < SEQ_LEN
        # tl.static_print("Backward - padding_mask: ", padding_mask)

        # Create the combined attention mask (Flash Attention + Sliding Attention + Attention Sinks)
        check_sink = k_offsets[None, :] < SINK_SIZE
        check_window = k_offsets[None, :]> (q_offsets[:, None] - WINDOW_SIZE)
        causal_mask = k_offsets[None, :] <= q_offsets[:, None]
        combined_mask = causal_mask & (check_sink | check_window)
        combined_mask &= padding_mask[None, :]
        # tl.static_print("Backward - combined_mask: ", combined_mask)

        # Load K and V blocks
        k_ptrs = K_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h + \
                 (k_offsets[None, :] * k_stride_s + tl.arange(0, HEAD_DIM)[:, None])
        v_ptrs = V_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h + \
                 (k_offsets[:, None] * v_stride_s + tl.arange(0, HEAD_DIM)[None, :])
        
        k_block = tl.load(k_ptrs, mask=padding_mask[None, :], other=0.0).to(tl.float32)
        v_block = tl.load(v_ptrs, mask=padding_mask[:, None], other=0.0).to(tl.float32)
        # tl.static_print("Backward - k_block, v_block: ", k_block, v_block)

        # Recompute attention scores (S) and probabilities (P)
        s_ij = tl.dot(q_block, k_block) * softmax_scale
        s_ij = tl.where(combined_mask, s_ij, -float('inf'))
        #  tl.static_print("Backward - s_ij: ", s_ij.shape)

        p_ij = tl.exp(s_ij - m_i[:, None])
        # tl.static_print("Backward - p_ij: ", p_ij)

        #   3. Use delta + dO to accumulate gradients for dq, dk, dv
        # Compute gradient of scores (dS)
        dS_ij_scaled = p_ij * (tl.dot(do_block, tl.trans(v_block)) - delta_i[:, None])

        # Accumulate dQ
        dq_acc += tl.dot(dS_ij_scaled, tl.trans(k_block)) * softmax_scale

        # Compute and atomically add dK
        dk_add = tl.dot(tl.trans(dS_ij_scaled), q_block) * softmax_scale
        dk_ptrs = dK_ptr + batch_idx * dk_stride_b + kv_head_idx * dk_stride_h + k_offsets[:, None] * dk_stride_s + tl.arange(0, HEAD_DIM)[None, :]
        # tl.static_print("Backward - dk_ptrs, dk_add: ", dk_ptrs, dk_add)
        tl.atomic_add(dk_ptrs, dk_add, mask=padding_mask[:, None])

        # Compute and atomically add dV
        dv_add = tl.dot(tl.trans(p_ij.to(do_block.dtype)), do_block)
        dv_ptrs = dV_ptr + batch_idx * dv_stride_b + kv_head_idx * dv_stride_h + k_offsets[:, None] * v_stride_s + tl.arange(0, HEAD_DIM)[None, :]
        # tl.static_print("Backward - dv_ptrs, dv_add: ", dv_ptrs, dv_add)
        tl.atomic_add(dv_ptrs, dv_add, mask=padding_mask[:, None])

    # Store the final dQ block
    dq_ptrs = dQ_ptr + batch_idx * dq_stride_b + q_head_idx * dq_stride_h + \
              (q_offsets[:, None] * dq_stride_s + tl.arange(0, HEAD_DIM)[None, :])
    tl.store(dq_ptrs, dq_acc.to(dQ_ptr.dtype.element_ty), mask=q_mask[:, None])

class FlashSWDAWithSink(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, window_size, sink_size, is_causal=True, softmax_scale=None):
        assert is_causal, "Currently, only causal attention is supported"

        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(q.shape[-1])

        batch, n_q_heads, seq_len, head_dim = q.shape
        _, n_kv_heads, _, _ = k.shape

        assert q.shape[0] == v.shape[0] and q.shape[2] == v.shape[2] and q.shape[3] == v.shape[3], "Query and Value shapes must be compatible except for num_heads"
        assert k.shape[0] == v.shape[0] and k.shape[1] == v.shape[1] and k.shape[2] == v.shape[2] and k.shape[3] == v.shape[3], "Key and Value shapes must be the same"
        assert head_dim <= 128, "Head dimension must be less than or equal to 128"
        assert n_q_heads % n_kv_heads == 0, "Number of query heads must be divisible by number of K/V heads"

        o = torch.empty_like(q)
        M = torch.empty((batch, n_q_heads, seq_len), device=q.device, dtype=torch.float32)


        BLOCK_M, BLOCK_N = 128, 64
        grid = (math.ceil(seq_len / BLOCK_M), batch * n_q_heads)

        _flash_attention_forward_swa_kernel[grid](
            q, k, v, o, M,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            M.stride(0), M.stride(1), M.stride(2),
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

        ctx.save_for_backward(q, k, v, o, M)
        ctx.softmax_scale = softmax_scale
        ctx.window_size = window_size
        ctx.sink_size = sink_size
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, M = ctx.saved_tensors
        softmax_scale = ctx.softmax_scale
        window_size = ctx.window_size
        sink_size = ctx.sink_size

        batch, n_q_heads, seq_len, head_dim = q.shape
        n_kv_heads = k.shape[1]

        dq = torch.empty_like(q, dtype=torch.float32)
        dk = torch.zeros_like(k, dtype=torch.float32)
        dv = torch.zeros_like(v, dtype=torch.float32)
        
        # TODO: Add your backward kernel here
        # Same initialization as other problem
        BLOCK_M, BLOCK_N = 128, 64
        grid = (math.ceil(seq_len / BLOCK_M), batch * n_q_heads)

        _flash_attention_backward_swa_kernel[grid](
            q, k, v, o, do, M,
            dq, dk, dv,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            do.stride(0), do.stride(1), do.stride(2),
            M.stride(0), M.stride(1), M.stride(2),
            dq.stride(0), dq.stride(1), dq.stride(2),
            dk.stride(0), dk.stride(1), dk.stride(2),
            dv.stride(0), dv.stride(1), dv.stride(2),
            softmax_scale,
            batch,
            n_q_heads,
            n_kv_heads,
            seq_len,
            WINDOW_SIZE=window_size,
            SINK_SIZE=sink_size,
            HEAD_DIM=head_dim,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )

        return dq, dk.to(k.dtype), dv.to(v.dtype), None, None, None, None
    
def flash_swda_with_sink(q, k, v, window_size: int, sink_size: int = 0, is_causal: bool = True, scale: Optional[float] = None):
    return FlashSWDAWithSink.apply(q, k, v, window_size, sink_size, is_causal, scale)