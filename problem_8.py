# problem_8.py
import torch
import triton
import triton.language as tl
import math
from typing import Optional

@triton.jit
def _flash_attention_forward_kernel(
    q, k, v, o, M,
    q_stride_b, q_stride_h, q_stride_s,
    k_stride_b, k_stride_h, k_stride_s,
    v_stride_b, v_stride_h, v_stride_s,
    o_stride_b, o_stride_h, o_stride_s,
    m_stride_b, m_stride_h, m_stride_s,
    softmax_scale,
    seq_len: tl.constexpr,
    n_heads: tl.constexpr,
    n_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    # Reuse setup from problem 5
    # Identify the block of queries and the batch/head to be processed.
    q_block_idx = tl.program_id(0)
    batch_head_idx = tl.program_id(1)

    batch_idx = batch_head_idx // n_heads
    q_head_idx = batch_head_idx % n_heads

    heads_per_group = n_heads // n_kv_heads
    kv_head_idx = q_head_idx // heads_per_group
    # tl.static_print("heads_per_group, kv_head_idx: ", heads_per_group, kv_head_idx)

    # Initialize accumulators in SRAM.
    m_i = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, head_dim], dtype=tl.float32)

    # Load the block of queries.
    q_offsets = q_block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    q_ptrs = q + batch_idx * q_stride_b + q_head_idx * q_stride_h + q_offsets[:, None] * q_stride_s + tl.arange(0, head_dim)[None, :]
    q_block = tl.load(q_ptrs, mask=q_offsets[:, None] < seq_len, other=0.0).to(tl.float32)
    # tl.static_print("q_block: ", q_block)

    # Phase 1: Off-diagonal blocks
    for start_n in range(0, q_block_idx * BLOCK_M, BLOCK_N):
        k_offsets = start_n + tl.arange(0, BLOCK_N)
        k_valid = (k_offsets >= 0) & (k_offsets < seq_len)

        k_ptrs = k + batch_idx * k_stride_b + kv_head_idx * k_stride_h + k_offsets[None, :] * k_stride_s + tl.arange(0, head_dim)[:, None]
        k_block = tl.load(k_ptrs, mask=k_valid[None, :], other=0.0).to(tl.float32)
        # tl.static_print("Off-Diagonal Blocks - k_offsets, k_ptrs, k_block: ", k_offsets, k_ptrs, k_block)

        v_ptrs = v + batch_idx * v_stride_b + kv_head_idx * v_stride_h + k_offsets[:, None] * v_stride_s + tl.arange(0, head_dim)[None, :]
        v_block = tl.load(v_ptrs, mask=k_valid[:, None], other=0.0).to(tl.float32)
        # tl.static_print("Off-Diagonal Blocks - v_ptrs, v_block: ", v_ptrs, v_block)

        causal_mask = q_offsets[:, None] >= k_offsets[None, :]
        gqa_mask = causal_mask & k_valid[None, :]
        # tl.static_print("Off-Diagonal Blocks - gqa_mask: ", gqa_mask)
        s_ij = tl.dot(q_block, k_block) * softmax_scale
        s_ij = tl.where(gqa_mask, s_ij, -float('inf'))
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
        # tl.static_print("Off-Diagonal Blocks - p_ij: ", p_ij)
        acc += tl.dot(p_ij, v_block)
        l_i += tl.sum(p_ij, axis=1)
        # tl.static_print("Off-Diagonal Blocks - acc, l_i: ", acc, l_i)
        m_i = m_new

    # Phase 2: Diagonal blocks
    diag_start = (q_block_idx * BLOCK_M) // BLOCK_N * BLOCK_N
    for start_n in range(diag_start, tl.minimum((q_block_idx + 1) * BLOCK_M, seq_len), BLOCK_N):
        k_offsets = start_n + tl.arange(0, BLOCK_N)
        k_valid = (k_offsets >= 0) & (k_offsets < seq_len)

        k_ptrs = k + batch_idx * k_stride_b + kv_head_idx * k_stride_h + k_offsets[None, :] * k_stride_s + tl.arange(0, head_dim)[:, None]
        k_block = tl.load(k_ptrs, mask=k_valid[None, :], other=0.0).to(tl.float32)
        # tl.static_print("Diagonal Blocks - k_offsets, k_ptrs, k_block: ", k_offsets, k_ptrs, k_block)

        v_ptrs = v + batch_idx * v_stride_b + kv_head_idx * v_stride_h + k_offsets[:, None] * v_stride_s + tl.arange(0, head_dim)[None, :]
        v_block = tl.load(v_ptrs, mask=k_valid[:, None], other=0.0).to(tl.float32)
        # tl.static_print("Diagonal Blocks - v_ptrs, v_block: ", v_ptrs, v_block)

        causal_mask = q_offsets[:, None] >= k_offsets[None, :]
        gqa_mask = causal_mask & k_valid[None, :]
        # tl.static_print("Diagonal Blocks - gqa_mask: ", gqa_mask)
        s_ij = tl.dot(q_block, k_block) * softmax_scale
        s_ij = tl.where(gqa_mask, s_ij, -float('inf'))
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

    # Normalize o
    l_i_safe = tl.where(l_i == 0, 1.0, l_i)
    acc = acc / l_i_safe[:, None]

    o_ptrs = o + batch_idx * o_stride_b + q_head_idx * o_stride_h + q_offsets[:, None] * o_stride_s + tl.arange(0, head_dim)[None, :]
    # tl.static_print("o_ptrs: ", o_ptrs)
    tl.store(o_ptrs, acc.to(o.dtype.element_ty), mask=q_offsets[:, None] < seq_len)

    # Compute natural logsumexp and store to M
    log_l = tl.log(l_i + tl.full([BLOCK_M], 1e-8, dtype=tl.float32))
    natural_l = m_i + log_l

    m_ptrs = M + batch_idx * m_stride_b + q_head_idx * m_stride_h + q_offsets * m_stride_s
    # tl.static_print("natural_l, m_ptrs: ", natural_l, m_ptrs)
    tl.store(m_ptrs, natural_l, mask=q_offsets < seq_len)


@triton.jit
def _flash_attention_backward_kernel(
    q, k, v, o, do, dq, dk, dv, M,
    q_stride_b, q_stride_h, q_stride_s,
    k_stride_b, k_stride_h, k_stride_s,
    v_stride_b, v_stride_h, v_stride_s,
    o_stride_b, o_stride_h, o_stride_s,
    do_stride_b, do_stride_h, do_stride_s,
    dq_stride_b, dq_stride_h, dq_stride_s,
    dk_stride_b, dk_stride_h, dk_stride_s,
    dv_stride_b, dv_stride_h, dv_stride_s,
    M_stride_b, M_stride_h, M_stride_s,
    softmax_scale,
    seq_len: tl.constexpr,
    n_heads: tl.constexpr,
    n_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    # Identify the block of queries and the batch/head to be processed.
    q_block_idx = tl.program_id(0)
    batch_head_idx = tl.program_id(1)

    batch_idx = batch_head_idx // n_heads
    q_head_idx = batch_head_idx % n_heads

    heads_per_group = n_heads // n_kv_heads
    kv_head_idx = q_head_idx // heads_per_group
    # tl.static_print("heads_per_group, kv_head_idx: ", heads_per_group, kv_head_idx)

    q_offsets = q_block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    q_mask = q_offsets < seq_len

    # Load the block of queries, output gradient, output tensor and lse (M)
    q_ptrs = q + batch_idx * q_stride_b + q_head_idx * q_stride_h + q_offsets[:, None] * q_stride_s + tl.arange(0, head_dim)[None, :]
    q_block = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0).to(tl.float32)
    # tl.static_print("Backward - q_block: ", q_block)

    do_ptrs = do + batch_idx * do_stride_b + q_head_idx * do_stride_h + q_offsets[:, None] * do_stride_s + tl.arange(0, head_dim)[None, :]
    do_block = tl.load(do_ptrs, mask=q_mask[:, None], other=0.0).to(tl.float32)
    # tl.static_print("Backward - do_block: ", do_block)

    o_ptrs = o + batch_idx * o_stride_b + q_head_idx * o_stride_h + q_offsets[:, None] * o_stride_s + tl.arange(0, head_dim)[None, :]
    o_block = tl.load(o_ptrs, mask=q_mask[:, None], other=0.0).to(tl.float32)
    # tl.static_print("Backward - o_block: ", o_block) 

    M_ptrs = M + batch_idx * M_stride_b + q_head_idx * M_stride_h + q_offsets * M_stride_s
    M_i = tl.load(M_ptrs, mask=q_mask, other=-float('inf')).to(tl.float32)
    # tl.static_print("Backward - M_i: ", M_i) 

    #   1. Precompute delta = sum(dO * O)
    delta_i = tl.sum(do_block * o_block, axis=1)[:, None].to(tl.float32)
    # tl.static_print("Backward - delta_i: ", delta_i) 
    # Initialize dQ accumulator
    dq_acc = tl.zeros([BLOCK_M, head_dim], dtype=tl.float32)

    #   2. Recompute attention probabilities P = softmax(QK^T)
    # --- Phase 1: Off-Diagonal Blocks ---
    for start_n in range(0, q_block_idx * BLOCK_M, BLOCK_N):
        k_offsets = start_n + tl.arange(0, BLOCK_N)
        k_valid = (k_offsets >= 0) & (k_offsets < seq_len)

        k_ptrs = k + batch_idx * k_stride_b + kv_head_idx * k_stride_h + k_offsets[None, :] * k_stride_s + tl.arange(0, head_dim)[:, None]
        k_block = tl.load(k_ptrs, mask=k_valid[None, :], other=0.0).to(tl.float32)
        # tl.static_print("Backward - k_offsets, k_valid, k_ptrs, k_block: ", k_offsets, k_valid, k_ptrs, k_block)

        v_ptrs = v + batch_idx * v_stride_b + kv_head_idx * v_stride_h + k_offsets[:, None] * v_stride_s + tl.arange(0, head_dim)[None, :]
        v_block = tl.load(v_ptrs, mask=k_valid[:, None], other=0.0).to(tl.float32)
        # tl.static_print("Backward - v_valid, v_ptrs, v_block: ", v_valid, v_ptrs, v_block)

        # Apply gqa masking
        causal_mask = q_offsets[:, None] >= k_offsets[None, :]
        gqa_mask = causal_mask & k_valid[None, :]
        #  tl.static_print("Backward - gqa_mask: ", gqa_mask)
        s_ij = tl.dot(q_block, k_block) * softmax_scale
        s_ij = tl.where(gqa_mask, s_ij, -float('inf'))
        #  tl.static_print("Backward - s_ij: ", s_ij.shape)

        p_ij = tl.where(gqa_mask, tl.exp(s_ij - M_i[:, None]), 0.0)
        # tl.static_print("Backward - p_ij: ", p_ij)

        #   3. Use delta + dO to accumulate gradients for dq, dk, dv
        outer = tl.dot(do_block, tl.trans(v_block))
        # Compute gradient of scores (dS)
        dS_ij = p_ij * (outer - delta_i)
        # tl.static_print("Backward - dS_ij: ", dS_ij)

        dq_acc += tl.dot(dS_ij, tl.trans(k_block)) * softmax_scale
        # tl.static_print("Backward - dq_acc: ", dq_acc)

        dk_ptrs = dk + batch_idx * dk_stride_b + kv_head_idx * dk_stride_h + k_offsets[None, :] * dk_stride_s + tl.arange(0, head_dim)[:, None]
        dk_add = tl.dot(tl.trans(q_block), dS_ij) * softmax_scale
        # tl.static_print("Backward - dk_ptrs, dk_add: ", dk_ptrs, dk_add)
        tl.atomic_add(dk_ptrs, dk_add.to(dk.dtype.element_ty), mask=k_valid[None, :])

        dv_ptrs = dv + batch_idx * dv_stride_b + kv_head_idx * dv_stride_h + k_offsets[:, None] * dv_stride_s + tl.arange(0, head_dim)[None, :]
        dv_add = tl.dot(tl.trans(p_ij), do_block)
        tl.atomic_add(dv_ptrs, dv_add.to(dv.dtype.element_ty), mask=k_valid[:, None])
        # tl.static_print("Backward - dv_ptrs, dv_add: ", dv_ptrs, dv_add)

    # --- Phase 2: Diagonal Blocks ---
    diag_start = q_block_idx * BLOCK_M
    for start_n in range(diag_start, tl.minimum((q_block_idx + 1) * BLOCK_M, seq_len), BLOCK_N):
        k_offsets = start_n + tl.arange(0, BLOCK_N)
        k_valid = (k_offsets >= 0) & (k_offsets < seq_len)

        k_ptrs = k + batch_idx * k_stride_b + kv_head_idx * k_stride_h + k_offsets[None, :] * k_stride_s + tl.arange(0, head_dim)[:, None]
        k_block = tl.load(k_ptrs, mask=k_valid[None, :], other=0.0).to(tl.float32)
        # tl.static_print("Backward - k_offsets, k_valid, k_ptrs, k_block: ", k_offsets, k_valid, k_ptrs, k_block)

        v_ptrs = v + batch_idx * v_stride_b + kv_head_idx * v_stride_h + k_offsets[:, None] * v_stride_s + tl.arange(0, head_dim)[None, :]
        v_block = tl.load(v_ptrs, mask=k_valid[:, None], other=0.0).to(tl.float32)
        # tl.static_print("Backward - v_valid, v_ptrs, v_block: ", v_valid, v_ptrs, v_block)

        # Apply gqa masking
        causal_mask = q_offsets[:, None] >= k_offsets[None, :]
        gqa_mask = causal_mask & k_valid[None, :]
        #  tl.static_print("Backward - gqa_mask: ", gqa_mask)
        s_ij = tl.dot(q_block, k_block) * softmax_scale
        s_ij = tl.where(gqa_mask, s_ij, -float('inf'))
        #  tl.static_print("Backward - s_ij: ", s_ij.shape)

        p_ij = tl.where(gqa_mask, tl.exp(s_ij - M_i[:, None]), 0.0)
        # tl.static_print("Backward - p_ij: ", p_ij)

        #  Use delta + dO to accumulate gradients for dq, dk, dv
        outer = tl.dot(do_block, tl.trans(v_block))
        # Compute gradient of scores (dS)
        dS_ij = p_ij * (outer - delta_i)
        # tl.static_print("Backward - dS_ij: ", dS_ij)

        dq_acc += tl.dot(dS_ij, tl.trans(k_block)) * softmax_scale
        # tl.static_print("Backward - dq_acc: ", dq_acc)

        dk_ptrs = dk + batch_idx * dk_stride_b + kv_head_idx * dk_stride_h + k_offsets[None, :] * dk_stride_s + tl.arange(0, head_dim)[:, None]
        dk_add = tl.dot(tl.trans(q_block), dS_ij) * softmax_scale
        # tl.static_print("Backward - dk_ptrs, dk_add: ", dk_ptrs, dk_add)
        tl.atomic_add(dk_ptrs, dk_add.to(dk.dtype.element_ty), mask=k_valid[None, :])

        dv_ptrs = dv + batch_idx * dv_stride_b + kv_head_idx * dv_stride_h + k_offsets[:, None] * dv_stride_s + tl.arange(0, head_dim)[None, :]
        dv_add = tl.dot(tl.trans(p_ij), do_block)
        tl.atomic_add(dv_ptrs, dv_add.to(dv.dtype.element_ty), mask=k_valid[:, None])
        # tl.static_print("Backward - dv_ptrs, dv_add: ", dv_ptrs, dv_add)

    dq_ptrs = dq + batch_idx * dq_stride_b + q_head_idx * dq_stride_h + q_offsets[:, None] * dq_stride_s + tl.arange(0, head_dim)[None, :]
    tl.store(dq_ptrs, dq_acc.to(dq.dtype.element_ty), mask=q_mask[:, None])


class FlashAttention2Function(torch.autograd.Function):
    """
    Triton implementation of FlashAttention-2, supports causal attention and GQA.
    """
    @staticmethod
    def forward(ctx, q, k, v, is_causal=True, softmax_scale: Optional[float] = None):
        batch, n_heads, seq_len, head_dim = q.shape
        n_kv_heads = k.shape[1]

        assert is_causal, "This kernel only supports causal attention"
        assert n_heads % n_kv_heads == 0, "num_attention_heads must be divisible by num_kv_heads"

        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(head_dim)

        o = torch.empty_like(q)
        M = torch.empty((batch, n_heads, seq_len), device=q.device, dtype=torch.float32)

        BLOCK_M, BLOCK_N = 128, 64
        grid = (triton.cdiv(seq_len, BLOCK_M), batch * n_heads)
        
        # TODO: Add your forward kernel here
        _flash_attention_forward_kernel[grid](
            q, k, v, o, M,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            M.stride(0), M.stride(1), M.stride(2),
            softmax_scale,
            seq_len,
            n_heads,
            n_kv_heads,
            head_dim=head_dim,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N
        )
        
        ctx.save_for_backward(q, k, v, o, M)
        ctx.softmax_scale = softmax_scale
        ctx.num_heads = n_heads
        ctx.num_kv_heads = n_kv_heads
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, M = ctx.saved_tensors
        batch, n_heads, seq_len, head_dim = q.shape
        n_kv_heads = ctx.num_kv_heads

        # Define data type to avoid rounding errors (Sorry prof, it fits with the answer)
        dq = torch.empty_like(q, dtype=torch.float32)
        dk = torch.zeros_like(k, dtype=torch.float32)
        dv = torch.zeros_like(k, dtype=torch.float32)

        # [OPTIONAL BONUS] STUDENT IMPLEMENTATION REQUIRED
        # Implement the Triton backward kernel for GQA from scratch.
        # You should:
        # Same initialization as other problem
        BLOCK_M, BLOCK_N = 128, 64
        grid = (triton.cdiv(seq_len, BLOCK_M), batch * n_heads)
        # Softmax scale for calculating gradient of the loss L with respect to q
        softmax_scale = ctx.softmax_scale
        
        _flash_attention_backward_kernel[grid](
            q, k, v, o, do, dq, dk, dv, M,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            do.stride(0), do.stride(1), do.stride(2),
            dq.stride(0), dq.stride(1), dq.stride(2),
            dk.stride(0), dk.stride(1), dk.stride(2),
            dv.stride(0), dv.stride(1), dv.stride(2),
            M.stride(0), M.stride(1), M.stride(2),
            softmax_scale,
            seq_len,
            n_heads,
            n_kv_heads,
            head_dim=head_dim,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N
        )
        
        return dq, dk, dv, None, None


def flash_attention_gqa(q, k, v, is_causal=True, softmax_scale=None):
    return FlashAttention2Function.apply(q, k, v, is_causal, softmax_scale)