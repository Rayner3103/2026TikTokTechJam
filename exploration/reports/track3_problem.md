# Track 3: GPU Kernel for Transformer Layer - Problem Understanding

## Problem Statement

The GPU Kernel track requires participants to **implement a custom GPU kernel** for a Transformer layer — specifically the self-attention mechanism — and optimize it for maximum throughput on modern GPUs. This is a low-level systems programming challenge that tests understanding of GPU architecture, memory hierarchies, and parallel computation.

## Core Problem

The fundamental problem is **high-performance GPU kernel implementation**:

1. **Attention Computation**: Implement the core attention mechanism (QK^T scaling, softmax, AV) efficiently on GPU
2. **Memory Optimization**: Minimize HBM access, maximize shared memory usage, optimize memory coalescing
3. **Parallel Execution**: Properly map attention computation to GPU thread blocks and warps
4. **Numerical Stability**: Handle softmax overflow, maintain precision with mixed precision
5. **Performance Tuning**: Achieve measurable speedup over baseline PyTorch/FlashAttention implementations
6. **Correctness**: Kernel outputs must match reference implementation within numerical tolerance

## Key Technical Domains

### Transformer Layer Components
- **Self-Attention**: Q, K, V projection; scaled dot-product attention
- **Multi-Head Attention**: Parallel attention heads; output projection
- **Add & Norm**: Residual connections; layer normalization
- **FFN**: Feed-forward network with activation and projection
- **Causal Masking** (if applicable): Mask future tokens for autoregressive generation

### GPU Architecture Considerations
- **Memory Hierarchy**: HBM → Shared Memory → L1/L2 Cache → Registers
- **Thread Hierarchy**: Thread → Warp (32 threads) → Block (up to 1024 threads) → Grid
- **Memory Access Patterns**: Coalesced access, bank conflicts, cache utilization
- **Precision Options**: FP32, FP16, BF16, INT8 — trade-offs between speed and accuracy
- **Tiling/Blocking**: Breaking large matrices into tiles for shared memory reuse

### Performance Metrics
- **Throughput**: Tokens per second (tokens/s) or MFU (Model FLOPs Utilization)
- **Latency**: Time per forward pass (ms)
- **Memory Bandwidth**: GB/s achieved vs. theoretical maximum
- **FLOPs Utilization**: % of theoretical peak FLOPs achieved
- **Occupancy**: % of active warps per SM

## Constraints

- **72-hour challenge**: GPU kernel development requires iterative testing and optimization
- **GPU availability**: Must test on available GPU hardware (likely NVIDIA A100/V100)
- **Correctness over speed**: Kernel must produce correct results before optimizing
- **Baseline comparison**: Must compare against PyTorch reference and potentially FlashAttention
- **Debugging difficulty**: GPU kernel debugging requires specialized tools (Nsight, pdb, print-based)

## Assumptions Made

1. NVIDIA GPU with CUDA 12.x is available (A100 or similar)
2. PyTorch with CUDA support is available for reference implementations
3. Triton (OpenAI's GPU language) may be an option if CUDA is too low-level
4. The kernel should support standard transformer dimensions (hidden_size, num_heads, seq_len)
5. "Transformer layer" likely means the full forward pass (attention + FFN + norm)
6. Batch size may be 1 (inference) or variable (training) — need to clarify

## What the Starter Kit Provides

- React UI for monitoring
- Fastify API for job submission
- Codex CLI for code assistance
- Docker containers for isolated GPU execution
- JSON persistence for benchmark results

## Key Technical Questions

1. What is the exact transformer layer variant (standard, causal, linear, rotary)?
2. What precision is targeted (FP32, FP16, BF16, mixed)?
3. What sequence lengths are the primary targets (short, long, variable)?
4. Should the kernel support batching or just single sequence?
5. Is FlashAttention-style memory optimization required, or standard attention?
6. What comparison baselines are expected (PyTorch vanilla, FlashAttention, etc.)?

## Implementation Approaches

### Option A: Pure CUDA C++
- Maximum performance potential
- Requires deep CUDA knowledge
- Hardest to implement and debug
- Best for competitive benchmarks

### Option B: Triton (Python-based GPU programming)
- Easier to implement and iterate
- Good performance (close to CUDA)
- Growing ecosystem (FlashAttention written in Triton)
- May have limitations for advanced optimizations

### Option C: PyTorch Custom Autograd
- Easiest to implement
- Limited optimization potential
- Good for prototyping
- May not meet performance requirements

## Performance Optimization Techniques

1. **Tiled Attention**: Process attention in tiles to maximize shared memory reuse
2. **Fused Operations**: Combine multiple operations into single kernel (e.g., QKV + attention)
3. **Shared Memory**: Cache K and V tiles to avoid repeated HBM reads
4. **Memory Coalescing**: Ensure threads in a warp access contiguous memory
5. **Register Tiling**: Keep frequently-accessed data in registers
6. **Warp-Level Primitives**: Use __shfl_sync for intra-warp communication
7. **Prefetching**: Overlap computation with memory transfers
8. **Persistent Kernels**: Keep GPU busy across multiple sequences

## Evaluation Criteria (Likely)

- **Correctness**: Output matches reference within tolerance (e.g., max error < 1e-3)
- **Speedup**: % improvement over PyTorch baseline
- **MFU**: Model FLOPs Utilization percentage
- **Scalability**: Performance across different sequence lengths and batch sizes
- **Code Quality**: Readability, comments, structure
- **Documentation**: Explanation of design choices and optimization decisions
