# Track 3 Transformer Throughput Optimization Roadmap

## Current State Analysis

### Existing Directions (3 implemented)

1. **Direction 1: SDPA + torch.compile**
   - ✅ Uses `F.scaled_dot_product_attention` (auto-selects FlashAttention)
   - ✅ `torch.compile` fusion for LayerNorm, residuals, FFN
   - Expected: 2-3x on CUDA
   - Risk: Low (mathematically identical)

2. **Direction 2: QKV Fusion + SDPA + torch.compile**
   - ✅ Fused Q/K/V projection (1 GEMM instead of 3)
   - ✅ SDPA attention tiling
   - Expected: 1.5-2x attention phase → 1.3-1.7x total
   - Risk: Medium (masking edge cases)

3. **Direction 3: Fused Layer-Level Kernel + torch.compile max-autotune**
   - ✅ Custom fused forward pass minimizing graph breaks
   - ✅ `torch.compile(mode="max-autotune")`
   - ✅ Manual residual handling
   - Expected: 2.5-4x on CUDA
   - Risk: High (graph break debugging)

---

## Gap Analysis: What's Missing?

| Category | Current | Gap | Opportunity |
|----------|---------|-----|-------------|
| **Attention Pattern** | Dense quadratic O(N²) | No sparse/local attention | Sliding window, sparse patterns for long sequences |
| **Precision** | FP32/FP16 | No mixed-precision ops | FP8/BF16 kernel fusion, quantization-aware compute |
| **Memory Layout** | Standard tensors | No kernel-level memory opt | Fused memory writes, reduced intermediate allocation |
| **Batch Processing** | Standard batching | No dynamic/adaptive | Token-level batching, variable-length grouping |
| **Cache Usage** | Default | No L1/L2 optimization | Register tiling, shared memory pre-load |
| **FFN Pattern** | Standard MLPs | No specialized kernels | Fused gate operations, specialized activations |
| **Inference Flow** | Full sequence | No incremental decoding | KV cache reuse, rotary embeddings, paged attention |
| **Architecture Variant** | Vanilla transformer | No MQA/GQA support | Multi-query/grouped-query attention |

---

## Proposed New Directions (4 emerging optimizations)

### Direction 4: Fused Gate Linear Units + Precision Downcasting

**Focus:** FFN optimization + mixed-precision inference

**Why This Matters:**
- FFN layers consume 60%+ of transformer compute
- Current implementations do separate gate + projection operations
- Fusing reduces memory bandwidth and kernel overhead
- Mixed precision (fp8/bf16) can achieve 2-3x speedup with minimal accuracy loss

**Key Optimizations:**
- Fused GLU/GELU: single kernel for `(x @ W1) * GELU(x @ W2)` 
- BF16 compute for FFN layers with FP32 accumulation
- Reduced intermediate tensor allocations
- Better L1/L2 cache locality

**Expected Speedup:** 1.2-1.8x (mainly on FFN-heavy models)

**Implementation Priority:** Medium (requires kernel fusion knowledge)

---

### Direction 5: Multi-Query/Grouped-Query Attention (MQA/GQA) + KV Cache

**Focus:** Inference optimization through attention head sharing

**Why This Matters:**
- MQA/GQA reduces KV cache size by 8-16x (critical for long sequences)
- Single K/V heads shared across multiple Q heads
- Dominant in modern models (Llama 2/3, Mistral, Claude)
- Inference throughput bottlenecked by KV cache memory bandwidth

**Key Optimizations:**
- Shared K/V projection heads
- Fused rotary embedding application
- Paged attention-style KV cache management
- Batch-level streaming for variable-length inputs

**Expected Speedup:** 1.5-2.5x on inference workloads (depending on sequence length)

**Implementation Priority:** High (most applicable to production inference)

---

### Direction 6: Sliding Window Attention + Sparse Pattern Support

**Focus:** Long-sequence efficiency through local attention patterns

**Why This Matters:**
- Local window attention reduces O(N²) → O(N*W) where W is window size
- Emerging in models like Mistral, Phi (sliding window = 4k tokens)
- Causal attention + local window = 4-8x memory savings
- Enables much longer effective sequences with same compute budget

**Key Optimizations:**
- Sliding window masking in fused kernel
- Diagonal band attention pattern
- Block-sparse attention patterns
- Efficient mask application without full QK^T materialization

**Expected Speedup:** 2-4x for long sequences (N > 2048)

**Implementation Priority:** Medium-High (sequence length dependent)

---

### Direction 7: Activation Checkpointing + Dynamic Layer Skipping

**Focus:** Memory efficiency + selective computation

**Why This Matters:**
- Not all layers contribute equally to output quality
- Token pruning / early exit techniques can skip redundant computation
- Activation checkpointing trades compute for memory in training-like scenarios
- Inference can benefit from adaptive compute allocation

**Key Optimizations:**
- Confidence-based layer skipping (route tokens to subset of layers)
- Sequence length-aware layer routers
- Fused checkpointing + recompute kernels
- Entropy-based pruning decision points

**Expected Speedup:** 1.3-1.8x (highly model-dependent, requires fine-tuning)

**Implementation Priority:** Low-Medium (requires model architecture changes)

---

## Ranking by Impact vs. Effort

| Direction | Impact | Effort | Risk | Best For |
|-----------|--------|--------|------|----------|
| **Dir 4: Fused GLU + Precision** | ⭐⭐⭐ | ⭐⭐ | Low | Dense compute-bound scenarios |
| **Dir 5: MQA/GQA + KV Cache** | ⭐⭐⭐⭐ | ⭐⭐⭐ | Medium | Production inference, long sequences |
| **Dir 6: Sliding Window + Sparse** | ⭐⭐⭐⭐ | ⭐⭐⭐ | Medium | Long sequences (N > 2k) |
| **Dir 7: Dynamic Skipping** | ⭐⭐ | ⭐⭐⭐⭐ | High | Adaptive inference pipelines |

---

## Recommended Implementation Priority

### Phase 1 (High ROI, Medium Effort)
**Direction 5 (MQA/GQA + KV Cache)** — Most aligned with production inference patterns

### Phase 2 (Good ROI, Lower Effort)
**Direction 4 (Fused GLU + Precision)** — Practical for FFN-heavy models

### Phase 3 (Sequence Length Dependent)
**Direction 6 (Sliding Window + Sparse)** — Best for long-context applications

### Phase 4 (Research-Grade)
**Direction 7 (Dynamic Skipping)** — Requires architectural changes

---

## References & Further Reading

### FlashAttention Variants
- FlashAttention-3: CuTe DSL for Hopper GPUs (4-5x speedup on H100)
- Reference: Dao et al., ICLR 2024

### Attention Optimization Papers
- **Multi-Query Attention**: Shazeer et al. (Google)
- **Grouped-Query Attention**: Ainslie et al. (Google)
- **Sliding Window Attention**: Used in Mistral 7B, Mistral-Nemo

### Production Implementations
- **vLLM**: Paged attention + KV cache optimizations
- **TensorRT-LLM**: Fused kernels + multi-GPU optimizations
- **xFormers**: Memory-efficient attention variants

### Key Optimization Patterns
- **Tiling**: Reduce memory access by overlapping computation
- **Fusion**: Combine multiple operations into single kernels
- **Quantization**: Reduce precision for speed + memory
- **Streaming**: Process data in chunks to hide latency
- **Pipelining**: Overlap communication and computation

---

## Next Steps

1. **Benchmark current directions** on various sequence lengths
2. **Profile memory bandwidth** bottlenecks
3. **Identify best candidate** among proposed directions
4. **Implement and measure** against baselines
5. **Iterate on parameters** (window sizes, quantization levels, etc.)
