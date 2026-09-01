# Solution 1: Triton-Based Fused Attention Kernel

**Importance: ██████████ (10/10) — Best balance of performance and feasibility**
**Ease of Implementation: ████████░░ (8/10) — High feasibility with Triton**

## Problem Solved

Addresses **Gap 1 (CUDA/Triton Environment)** and **Gap 2 (Benchmarking)** — leveraging Triton's Python-based GPU programming to quickly implement and optimize an attention kernel with proper benchmarking.

## Concept

Build a **Triton-based fused attention kernel** that combines QKV projection and attention computation into a single kernel launch, minimizing memory transfers and maximizing GPU utilization. Triton provides the right balance of performance (close to CUDA) and productivity (Python-based, easy to iterate).

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│              Triton Fused Attention                      │
│                                                          │
│  ┌─────────────┐    ┌─────────────┐                    │
│  │   QKV       │    │  Attention  │                    │
│  │  Projection │───▶│   Kernel    │                    │
│  │  (Fused)    │    │             │                    │
│  └─────────────┘    │  ┌───────┐  │                    │
│                     │  │Q×K^T  │  │                    │
│                     │  │ Scale  │  │                    │
│                     │  │ Softmax│  │                    │
│                     │  │ ×V     │  │                    │
│                     │  └───────┘  │                    │
│                     └─────────────┘                    │
│                                                          │
│  ┌─────────────────────────────────┐                    │
│  │  Performance Optimizations      │                    │
│  │                                  │                    │
│  │  • Tiled attention              │                    │
│  │  • Shared memory for K,V        │                    │
│  │  • Fused softmax                │                    │
│  │  • Masked attention             │                    │
│  └─────────────────────────────────┘                    │
└─────────────────────────────────────────────────────────┘
```

## Technical Implementation

### 1. Triton Kernel Implementation
```python
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8),
    ],
    key=['N_CTX', 'HEAD_DIM'],
)
@triton.jit
def fused_attention_kernel(
    Q, K, V, sm_scale,
    Out,
    stride_q_batch, stride_q_seq, stride_q_head, stride_q_dim,
    stride_k_batch, stride_k_seq, stride_k_head, stride_k_dim,
    stride_v_batch, stride_v_seq, stride_v_head, stride_v_dim,
    stride_o_batch, stride_o_seq, stride_o_head, stride_o_dim,
    Batch, N_CTX, H, HEAD_DIM,
    causal: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Program ID
    program_id = tl.program_id(0)
    
    # Block indices
    block_m = program_id // N_CTX
    block_n = program_id % (N_CTX // BLOCK_SIZE_N)
    
    # Pointer to Q, K, V tiles
    q_ptrs = Q + block_m * BLOCK_SIZE_M * H * HEAD_DIM + tl.arange(0, BLOCK_SIZE_M)[:, None] * H * HEAD_DIM + tl.arange(0, HEAD_DIM)[None, :]
    k_ptrs = K + tl.arange(0, BLOCK_SIZE_N)[None, :] * H * HEAD_DIM + tl.arange(0, HEAD_DIM)[:, None]
    v_ptrs = V + tl.arange(0, BLOCK_SIZE_N)[:, None] * H * HEAD_DIM + tl.arange(0, HEAD_DIM)[None, :]
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, HEAD_DIM), dtype=tl.float32)
    
    # Softmax state
    m_i = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32)
    
    # K, V tiles
    for start_n in range(0, N_CTX, BLOCK_SIZE_K):
        k = tl.load(k_ptrs + start_n * H * HEAD_DIM)
        v = tl.load(v_ptrs + start_n * H * HEAD_DIM)
        
        # QK^T computation
        qk = tl.load(q_ptrs) * k * sm_scale
        
        # Mask for causal attention
        if causal:
            mask = tl.arange(0, BLOCK_SIZE_M)[:, None] >= (start_n + tl.arange(0, BLOCK_SIZE_K))[None, :]
            qk = tl.where(mask, qk, float('-inf'))
        
        # Softmax
        m_new = tl.maximum(tl.max(qk, 1), m_i)
        l_new = tl.exp(m_i - m_new) * l_i + tl.sum(tl.exp(qk - m_new[:, None]), 1)
        
        # Scale and accumulate
        acc = tl.exp(m_i - m_new)[:, None] * acc
        acc += tl.exp(qk - m_new[:, None]) * v
        
        m_i = m_new
        l_i = l_new
    
    # Final normalization
    acc = acc / l_i[:, None]
    
    # Write output
    o_ptrs = Out + block_m * BLOCK_SIZE_M * H * HEAD_DIM + tl.arange(0, BLOCK_SIZE_M)[:, None] * H * HEAD_DIM + tl.arange(0, HEAD_DIM)[None, :]
    tl.store(o_ptrs, acc)
```

### 2. Fused QKV Projection + Attention
```python
@triton.jit
def fused_qkv_attention_kernel(
    input, weight_q, weight_k, weight_v,
    sm_scale,
    output,
    stride_input_batch, stride_input_seq, stride_input_dim,
    stride_weight_q_head, stride_weight_q_dim, stride_weight_q_head_dim,
    ...
    Batch, SEQ_LEN, HIDDEN, NUM_HEADS, HEAD_DIM,
    causal: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Fused kernel that computes:
    1. Q = input @ W_Q
    2. K = input @ W_K
    3. V = input @ W_V
    4. Attention(Q, K, V)
    All in a single kernel launch
    """
    # Load input, compute Q/K/V in registers
    # Then compute attention on tiles
    # Write output
    pass
```

### 3. Benchmark Harness
```python
import torch
import time

class AttentionBenchmark:
    def __init__(self, device='cuda'):
        self.device = device
        self.baselines = {
            'pytorch_native': self.pytorch_native_attention,
            'torch_multihead': self.torch_multihead_attention,
        }
    
    def pytorch_native_attention(self, q, k, v, causal=False):
        """PyTorch scaled dot-product attention"""
        if causal:
            mask = torch.tril(torch.ones(q.size(-2), k.size(-2))).to(q.device)
            attn = torch.softmax((q @ k.transpose(-2, -1) / (q.size(-1) ** 0.5)) * mask, dim=-1)
        else:
            attn = torch.softmax(q @ k.transpose(-2, -1) / (q.size(-1) ** 0.5), dim=-1)
        return attn @ v
    
    def torch_multihead_attention(self, q, k, v, num_heads):
        """PyTorch MultiheadAttention"""
        attn_layer = torch.nn.MultiheadAttention(
            embed_dim=q.size(-1),
            num_heads=num_heads,
            batch_first=True,
        ).to(self.device)
        return attn_layer(q, k, v)[0]
    
    def benchmark_kernel(self, q, k, v, kernel_fn, name):
        """Benchmark a kernel with warmup and timing"""
        # Warmup
        for _ in range(10):
            kernel_fn(q, k, v)
        torch.cuda.synchronize()
        
        # Timing
        start = time.time()
        for _ in range(100):
            result = kernel_fn(q, k, v)
        torch.cuda.synchronize()
        elapsed = (time.time() - start) / 100
        
        # Verify correctness
        ref = self.pytorch_native_attention(q, k, v)
        error = torch.max(torch.abs(result - ref)).item()
        
        return {
            'kernel': name,
            'latency_ms': elapsed * 1000,
            'throughput_tokens': q.size(-2) / elapsed,
            'max_error': error,
            'passed': error < 1e-3,
        }
    
    def run_full_benchmark(self, batch_sizes, seq_lengths, head_dims):
        """Run benchmarks across configurations"""
        results = []
        
        for batch in batch_sizes:
            for seq in seq_lengths:
                for hd in head_dims:
                    q = torch.randn(batch, seq, 1, hd, device=self.device)
                    k = torch.randn(batch, seq, 1, hd, device=self.device)
                    v = torch.randn(batch, seq, 1, hd, device=self.device)
                    
                    for name, fn in {
                        'PyTorch Native': self.pytorch_native_attention,
                        'Triton Fused': self.triton_attention,
                        'PyTorch MHA': self.torch_multihead_attention,
                    }.items():
                        result = self.benchmark_kernel(q, k, v, fn, name)
                        result.update({
                            'batch_size': batch,
                            'seq_length': seq,
                            'head_dim': hd,
                        })
                        results.append(result)
        
        return results
```

### 4. Configuration-Driven Kernel
```python
class AttentionKernelConfig:
    def __init__(
        self,
        head_dim=64,
        num_heads=8,
        causal=True,
        precision='float16',
        use_tiling=True,
        tile_size=128,
    ):
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.causal = causal
        self.precision = precision
        self.use_tiling = use_tiling
        self.tile_size = tile_size
    
    def compile(self):
        """Compile the kernel with the given configuration"""
        # Select appropriate triton.Config based on config
        # Set precision-specific optimizations
        # Generate the kernel
        pass
```

## Justification

- **High Impact**: Triton is the right abstraction level for a hackathon — fast iteration, good performance
- **High Feasibility**: Triton is well-documented; FlashAttention reference provides starting point
- **Hackathon Friendly**: Can implement and iterate within 72 hours
- **Evaluatable**: Automated benchmark harness provides clear performance metrics
- **Scalable**: Easy to add optimizations (causal mask, different precisions, etc.)

## References

- Triton Tutorial: https://triton-lang.org/main/getting-started/tutorials/
- FlashAttention in Triton: https://github.com/openai/triton/blob/main/python/tutorials/06-fused-attention.py
- Triton Autotuning: https://triton-lang.org/main/getting-started/tutorials/03-fused-matmul.html
