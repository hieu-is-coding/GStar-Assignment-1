import sys
import argparse
import time
import math

import torch
import torch.nn.functional as F

DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

def create_mask_bool(
    seq_len: int,
    window_size: int,
    sink_size: int,
    device=None
    ) -> torch.Tensor:
    
    idx = torch.arange(seq_len, device=device)
    row = idx.unsqueeze(1)
    col = idx.unsqueeze(0)

    sliding = (col <= row) & (col >= row - (window_size - 1))
    sink = (col < sink_size) & (col <= row)

    return sliding | sink

def naive_attention(q, k, v, seq_len, window_size, sink_size):
    return F.scaled_dot_product_attention(
        query=q,
        key=k,
        value=v,
        attn_mask=create_mask_bool(seq_len, window_size, sink_size, device=q.device),
        enable_gqa=True,
    )

def benchmark_all_passes(triton_func, naive_func, test_params, problem_num):
    """
    Utility to benchmark the forward, backward, and combined passes of an attention function
    and compare it to a PyTorch implementation.
    """
    print("\n--- Running Performance Benchmark (Forward, Backward, Combined) ---")
    
    batch, heads_q, heads_kv, seq_len, dim, window_size, sink_size = test_params
    
    if problem_num == 8:
        config_str = f"B={batch}, Hq={heads_q}, Hkv={heads_kv}, L={seq_len}, D={dim}"
    elif problem_num == 9:
        config_str = f"B={batch}, Hq={heads_q}, Hkv={heads_kv}, L={seq_len}, D={dim}, W={window_size}, S={sink_size}"
    else:
        raise ValueError(f"Problem {problem_num} not supported for benchmarking")
        
    print(f"Benchmark Config: {config_str}")

    q_triton = torch.randn(batch, heads_q, seq_len, dim, device='cuda', dtype=DTYPE, requires_grad=True)
    k_triton = torch.randn(batch, heads_kv, seq_len, dim, device='cuda', dtype=DTYPE, requires_grad=True)
    v_triton = torch.randn(batch, heads_kv, seq_len, dim, device='cuda', dtype=DTYPE, requires_grad=True)
    
    q_ref = q_triton.clone().detach().requires_grad_(True)
    k_ref = k_triton.clone().detach().requires_grad_(True)
    v_ref = v_triton.clone().detach().requires_grad_(True)

    dout = torch.randn(batch, heads_q, seq_len, dim, device='cuda', dtype=DTYPE)

    def _run_benchmark(func, q, k, v):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        
        # Warm-up runs
        for _ in range(5):
            q.grad = k.grad = v.grad = None
            o = func()
            o.backward(dout, retain_graph=True)
        
        torch.cuda.synchronize()
        
        # Timed runs
        total_forward_time, total_backward_time = 0, 0
        num_runs = 20
        for _ in range(num_runs):
            q.grad = k.grad = v.grad = None
            
            torch.cuda.synchronize()
            start_forward = time.time()
            o = func()
            torch.cuda.synchronize()
            end_forward = time.time()
            
            total_forward_time += (end_forward - start_forward)
            
            torch.cuda.synchronize()
            start_backward = time.time()
            o.backward(dout, retain_graph=True)
            torch.cuda.synchronize()
            end_backward = time.time()
            
            total_backward_time += (end_backward - start_backward)

        avg_forward_ms = total_forward_time * 1000 / num_runs
        avg_backward_ms = total_backward_time * 1000 / num_runs
        avg_total_ms = (total_forward_time + total_backward_time) * 1000 / num_runs
        peak_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
        
        return avg_forward_ms, avg_backward_ms, avg_total_ms, peak_mem_gb

    if problem_num == 8:
        triton_wrapper = lambda: triton_func(q_triton, k_triton, v_triton, is_causal=True)
        naive_wrapper = lambda: naive_func(q_ref, k_ref, v_ref, seq_len=seq_len, window_size=seq_len, sink_size=0)
    elif problem_num == 9:
        triton_wrapper = lambda: triton_func(q_triton, k_triton, v_triton, window_size=window_size, sink_size=sink_size, is_causal=True)
        naive_wrapper = lambda: naive_func(q_ref, k_ref, v_ref, seq_len, window_size, sink_size)

    fwd_triton, bwd_triton, total_triton, mem_triton = _run_benchmark(triton_wrapper, q_triton, k_triton, v_triton)
    fwd_torch, bwd_torch, total_torch, mem_torch = _run_benchmark(naive_wrapper, q_ref, k_ref, v_ref)

    print("\n--- Benchmark Results (Avg Time in ms) ---")
    print(f"{'Implementation':<20}                  | {'Forward':<15} | {'Backward':<15} | {'Total':<15} | {'Peak Memory (GB)':<20}")
    print("-" * 90)
    print(f"{'PyTorch (Naive)':<20}                  | {fwd_torch:<15.4f} | {bwd_torch:<15.4f} | {total_torch:<15.4f} | {mem_torch:<20.4f}")
    print(f"{'Triton (GQA + SWDA + Attention Sinks)':<20} | {fwd_triton:<15.4f} | {bwd_triton:<15.4f} | {total_triton:<15.4f} | {mem_triton:<20.4f}")
    print("-" * 90)
    
    # Calculate speedups and memory savings
    fwd_speedup = fwd_torch / fwd_triton if fwd_triton > 0 else float('inf')
    bwd_speedup = bwd_torch / bwd_triton if bwd_triton > 0 else float('inf')
    total_speedup = total_torch / total_triton if total_triton > 0 else float('inf')
    mem_saving = mem_torch / mem_triton if mem_triton > 0 else float('inf')

    print(f"Forward Pass Speedup: Triton is {fwd_speedup:.2f}x faster.")
    print(f"Backward Pass Speedup: Triton is {bwd_speedup:.2f}x faster.")
    print(f"Overall Speedup: Triton is {total_speedup:.2f}x faster.")
    print(f"Memory Savings: Triton uses {mem_saving:.2f}x less memory.")
    
def check_backward_correctness(triton_func, problem_num):
    test_cases = [
        (1, 16, 16, 4096, 16, 256, 4),
        (1, 16, 8, 4096, 16, 256, 4),
        (1, 16, 1, 4096, 16, 256, 4),
    ]
    all_correct = True
    for case in test_cases:
        batch, heads_q, heads_kv, seq_len, dim, window_size, sink_size = case
        
        print("-" * 50)
        if problem_num == 8:
            print(f"Running test case: batch={batch}, heads_q={heads_q}, heads_kv={heads_kv}, seq_len={seq_len}, dim={dim}")
        elif problem_num == 9:
            print(f"Running test case: batch={batch}, heads_q={heads_q}, heads_kv={heads_kv}, seq_len={seq_len}, dim={dim}, window_size={window_size}, sink_size={sink_size}")
        else:
            raise ValueError(f"Problem {problem_num} not supported")
        
        q = torch.randn(batch, heads_q, seq_len, dim, device='cuda', dtype=DTYPE, requires_grad=True)
        k = torch.randn(batch, heads_kv, seq_len, dim, device='cuda', dtype=DTYPE, requires_grad=True)
        v = torch.randn(batch, heads_kv, seq_len, dim, device='cuda', dtype=DTYPE, requires_grad=True)
        
        q_ref, k_ref, v_ref = q.clone().detach().requires_grad_(), k.clone().detach().requires_grad_(), v.clone().detach().requires_grad_()
        
        if problem_num == 8:
            o_ref = naive_attention(q_ref, k_ref, v_ref, seq_len=seq_len, window_size=seq_len, sink_size=0)
            o_triton = triton_func(q, k, v, is_causal=True)
        elif problem_num == 9:
            o_ref = naive_attention(q_ref, k_ref, v_ref, seq_len, window_size, sink_size)
            o_triton = triton_func(q, k, v, window_size=window_size, sink_size=sink_size, is_causal=True)
        else:
            raise ValueError(f"Problem {problem_num} not supported")
            
        is_forward_correct = torch.allclose(o_ref, o_triton, atol=1e-2, rtol=1e-2)
        print(f"✅ Forward Pass Correctness: {'PASSED' if is_forward_correct else 'FAILED'}")
        
        dout = torch.rand_like(o_ref)
        o_ref.backward(dout)
        dq_ref, dk_ref, dv_ref = q_ref.grad, k_ref.grad, v_ref.grad
        
        o_triton.backward(dout)
        dq_flash, dk_flash, dv_flash = q.grad, k.grad, v.grad
        
        is_dq_correct = torch.allclose(dq_ref, dq_flash, atol=5e-2, rtol=5e-2)
        is_dk_correct = torch.allclose(dk_ref, dk_flash, atol=5e-2, rtol=5e-2)
        is_dv_correct = torch.allclose(dv_ref, dv_flash, atol=5e-2, rtol=5e-2)

        print(f"✅ Backward Pass dQ Correctness: {'PASSED' if is_dq_correct else 'FAILED'}")
        print(f"✅ Backward Pass dK Correctness: {'PASSED' if is_dk_correct else 'FAILED'}")
        print(f"✅ Backward Pass dV Correctness: {'PASSED' if is_dv_correct else 'FAILED'}")

        if not (is_forward_correct and is_dq_correct and is_dk_correct and is_dv_correct):
            all_correct = False

    if all_correct:
        print(f"\nAll P{problem_num} correctness tests passed!")
        benchmark_all_passes(triton_func, naive_attention, test_cases[-1], problem_num)


def check_problem_8():
    """Checks Problem 8: GQA."""
    problem_num = 8
    print(f"\n--- Running Autograder for Problem {problem_num}: GQA Backward Pass ---")
    try:
        from problem_8 import flash_attention_gqa
    except ImportError:
        print(f"Could not import FlashAttention2Function from solution_{problem_num}.py.")
        return
    
    torch.manual_seed(48)
    check_backward_correctness(flash_attention_gqa, problem_num)
    

def check_problem_9():
    """Checks Problem 9: GQA + SWDA + Attention Sinks Backward Pass."""
    problem_num = 9
    print(f"\n--- Running Autograder for Problem {problem_num}: GQA + SWDA + Attention Sinks Backward Pass ---")
    try:
        from problem_9 import flash_swda_with_sink
    except ImportError:
        print(f"Could not import FlashAttention2Function from solution_{problem_num}.py.")
        return
    
    torch.manual_seed(48)
    check_backward_correctness(flash_swda_with_sink, problem_num)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autograder for Triton Flash Attention assignments.")
    parser.add_argument('--p8', action='store_true', help='Run autograder for Problem 8 (GQA Backward Pass).')
    parser.add_argument('--p9', action='store_true', help='Run autograder for Problem 9 (GQA + SWDA + Attention Sinks Backward Pass).')
    
    if not torch.cuda.is_available():
        print("💥 CUDA not available. Skipping all GPU tests.")
        sys.exit(1)

    args = parser.parse_args()
    if not any(vars(args).values()):
        args.p8 = args.p9 = True
    
    if args.p8:
        check_problem_8()
    if args.p9:
        check_problem_9()