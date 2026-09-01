# New Directions Implementation Guide

## Summary

Created 4 new optimization directions (Directions 4-7) to explore additional throughput improvements beyond the original 3 directions. These complement the existing SDPA, QKV Fusion, and Layer-Level Fusion approaches.

---

## Direction 4: Fused GLU + Mixed-Precision Inference

**File:** [direction4/direction4_optimized.py](direction4/direction4_optimized.py)

### Key Features
- **Fused Gate Linear Unit**: Combines gate and value projections into single GEMM
  - Reduces 2 matrix multiplications to 1
  - Combines attention and FFN optimizations
- **Architecture Changes**:
  - `FusedGLU`: (x @ W1) * activation(x @ W2) fused
  - `FusedAttentionLayer`: Puts both attention + FFN optimization together
- **Integration**: Works with `torch.compile` for layer fusion

### Expected Performance
- **Speedup**: 1.2-1.8x on FFN-heavy architectures
- **Best For**: Models with large FFN dimensions (e.g., Llama 70B → 8B)
- **Precision**: FP32 LayerNorm + FP16 compute for FFN (safe accumulation)

### Configuration Parameters
```python
config.d_model = 128
config.num_heads = 8
config.ffn_dim = 512  # Or higher for better speedup
config.num_layers = 4
config.compile = True
```

### How It Works
```
Standard FFN:          Fused GLU:
x @ W_value           [x @ W_fused] → chunk → gated_value @ W_out
x @ W_gate   →        (1 GEMM + low-cost ops)
F.gelu(gate) * value

Memory access: 2 GEMM + 2 writes → 1 GEMM + 1 write
```

---

## Direction 5: Multi-Query/Grouped-Query Attention (MQA/GQA) + KV Cache

**File:** [direction5/direction5_optimized.py](direction5/direction5_optimized.py)

### Key Features
- **Grouped-Query Attention**: Share K/V heads across multiple Q heads
  - Reduces KV cache by 8-16x (single K/V heads vs multiple)
  - Critical for long-sequence inference
- **Rotary Embeddings (RoPE)**: Efficient position encoding without learnable params
- **KV Cache Optimization**: Pre-allocated cache for incremental decoding
- **Architecture Changes**:
  - `GroupedQueryAttention`: Q(num_heads) × K(num_kv_heads) × V(num_kv_heads)
  - `RotaryEmbedding`: Position-based rotation for relative distance
  - Cache return for inference phase

### Expected Performance
- **Speedup**: 1.5-2.5x on inference workloads
- **Memory Savings**: 8-16x reduction in KV cache for same sequence length
- **Best For**: Production inference, long sequences (N > 512)
- **Industry Adoption**: Used in Llama 2/3, Mistral, Claude

### Configuration Parameters
```python
config.d_model = 128
config.num_heads = 8
config.num_kv_heads = 2  # Key: reduces KV by 4x
config.ffn_dim = 512
config.num_layers = 4
config.max_seq_len = 2048
config.compile = True
```

### How It Works
```
Standard Attention:        Grouped-Query Attention:
Q: (b, 8 heads, s, 64)     Q: (b, 8 heads, s, 64)
K: (b, 8 heads, s, 64)     K: (b, 2 heads, s, 64) ← shared
V: (b, 8 heads, s, 64)     V: (b, 2 heads, s, 64) ← shared
                           → Repeat K/V for attention compute
                           → Cache only 1/4 the memory
```

---

## Direction 6: Sliding Window + Sparse Attention

**File:** [direction6/direction6_optimized.py](direction6/direction6_optimized.py)

### Key Features
- **Local Window Attention**: O(N²) → O(N * W) complexity
  - Each position attends to ±W tokens around it
  - Default: W=4096 (Mistral pattern)
- **Block-Sparse Patterns**: BigBird-style attention (local + strided + global)
  - Local window: essential attention
  - Strided: every k-th position
  - Global: first/last blocks (full attention)
- **Efficient Mask Computation**: Diagonal band construction

### Expected Performance
- **Speedup**: 2-4x for long sequences (N > 2048)
- **Memory**: Linear in sequence length vs quadratic
- **Best For**: Long-context tasks (documents, code, conversations)
- **Industry Adoption**: Mistral 7B, Phi-3, Grok

### Configuration Parameters
```python
config.d_model = 128
config.num_heads = 8
config.ffn_dim = 512
config.num_layers = 4
config.window_size = 4096  # Local window width
config.use_sparse_attention = False  # True for BigBird pattern
config.compile = True
```

### How It Works
```
Standard Attention:        Sliding Window:
All pairs computed          Only local window:
(N² operations)            
                           Query at pos i attends to
                           keys in [i-W, i+W]
                           (O(N*W) operations)

Long sequences:
N=16384, W=4096 → 16K×4K = 64M ops vs 16K²=256M ops (4x reduction)
```

---

## Direction 7: Adaptive Layer Skipping + Early Exit

**File:** [direction7/direction7_optimized.py](direction7/direction7_optimized.py)

### Key Features
- **Confidence-Based Routing**: Lightweight router decides layer execution
  - Per-token decision: skip entire layer or skip FFN only
  - Learned routing network guides skip decisions
- **Early Exit Heads**: Optional classifiers at each layer
  - Allows tokens to "exit early" if confident enough
  - Inspired by BranchyNet and Mixture-of-Experts literature
- **Soft Gating (Training) + Hard Gating (Inference)**:
  - Training: soft gates for smooth gradients
  - Inference: hard binary decisions for speed

### Expected Performance
- **Speedup**: 1.3-1.8x (model-dependent, requires fine-tuning)
- **Challenges**: Requires training-time optimization, curriculum learning
- **Best For**: Adaptive inference pipelines with variable compute budgets
- **Research Grade**: Experimental, not yet production-ready

### Configuration Parameters
```python
config.d_model = 128
config.num_heads = 8
config.ffn_dim = 512
config.num_layers = 4
config.use_early_exit = False  # Set True to enable early exit
config.exit_confidence_threshold = 0.95
config.compile = True
```

### How It Works
```
Router Network (lightweight):
  Input: hidden state → Linear(d→d/2) → ReLU → Linear(d/2→2)
  Outputs: [skip_layer_prob, skip_ffn_prob]

Token Processing:
  For each token at each layer:
    1. Router predicts: "skip this layer?" → prob 0.0-1.0
    2. Layer execution gates by (1 - skip_prob)
    3. If early_exit: predict class, exit if confident > threshold

Compute Reduction:
  Easy tokens: skip most layers (skip_prob → 1.0)
  Hard tokens: full computation (skip_prob → 0.0)
```

---

## Benchmarking New Directions

### Run Individual Direction
```bash
python3 run_all.py --start 4 --end 4  # Direction 4 only
python3 run_all.py --start 5 --end 5  # Direction 5 only
python3 run_all.py --start 6 --end 7  # Directions 6-7
```

### Run All Directions
```bash
python3 run_all.py  # Runs directions 1-7
```

### Expected Benchmark Output
Each direction tested on 3 configurations:
- **Small**: 4B × 128L (baseline)
- **Medium**: 8B × 512L (standard)
- **Large**: 16B × 1024L (stress test)

Per-config metrics:
- Baseline latency: mean/median/p95/p99 (ms)
- Optimized latency: same metrics
- Throughput: tokens/second
- Memory: peak usage and savings %
- Speedup: latency ratio
- Accuracy: abs/rel error vs baseline

---

## Performance Expectations Summary

| Direction | Focus | Expected Speedup | Sequence Dep | Training Impact |
|-----------|-------|------------------|--------------|-----------------|
| **1** | SDPA + compile | 2-3x | No | None (drop-in) |
| **2** | QKV Fusion | 1.5-2x | No | None (fused weights) |
| **3** | Layer Fusion | 2.5-4x | No | Compilation overhead |
| **4** | GLU + Precision | 1.2-1.8x | No | FFN-specific |
| **5** | MQA/GQA + Cache | 1.5-2.5x | **Yes ↑** | Retrain needed |
| **6** | Sliding Window | 2-4x | **Yes ↑↑** | Retrain needed |
| **7** | Layer Skipping | 1.3-1.8x | Variable | Major retraining |

**Key Observations:**
- Directions 1-4: Minimal changes, drop-in optimization
- Directions 5-7: Better speedups but require retraining or fine-tuning
- MQA/GQA (Dir 5): Best balance of speedup + practicality
- Sliding Window (Dir 6): Best for long sequences but needs retraining
- Layer Skipping (Dir 7): Experimental, highest complexity

---

## Integration with run_all.py

The benchmark script automatically discovers and tests all direction*/ folders:

```python
# Auto-discovery in run_all.py:
directions = discover_directions(start, end)  # Finds direction1/ ... direction7/

# For each direction:
impl_path = f"direction{n}/direction{n}_optimized.py"
Transformer = mod.OptimizedTransformer  # Loaded from each direction

# Tests on config variations:
CONFIGS = [
    {"name": "small", "batch": 4, "seq_len": 128, ...},
    {"name": "medium", "batch": 8, "seq_len": 512, ...},
    {"name": "large", "batch": 16, "seq_len": 1024, ...},
]
```

Results saved to `results.json` with detailed metrics for analysis.

---

## Next Steps

1. **Run Benchmarks**:
   ```bash
   python3 run_all.py
   ```

2. **Analyze Results**:
   - Compare speedups across directions and configs
   - Identify which direction performs best for each scenario
   - Look for memory/latency tradeoffs

3. **Profile Bottlenecks**:
   - Use NVIDIA Nsight Systems for kernel-level analysis
   - Identify memory bandwidth vs compute bound scenarios
   - Check for graph compilation overhead

4. **Selective Optimization**:
   - Combine best performers from each category
   - E.g., Direction 5 (MQA/GQA) + Direction 1 (SDPA)
   - Implement fusion of top performers

5. **Production Deployment**:
   - Profile on target hardware (A100, H100, etc.)
   - Measure end-to-end throughput vs latency tradeoff
   - Consider batch size, sequence length distributions

---

## References

### Papers & Implementations
- **FlashAttention-2**: Dao et al., ICLR 2024 (SDPA backend)
- **Multi-Query Attention**: Shazeer et al., Google (Direction 5 basis)
- **RoPE Embeddings**: Su et al., 2021 (Position encoding in Direction 5)
- **Sliding Window**: Mistral 7B, used in production models
- **Early Exit**: Teerapittayanon et al., "BranchyNet" (Direction 7 inspiration)
- **Mixture of Experts**: Shazeer et al. (routing mechanism reference)

### Production Systems
- **vLLM**: Efficient inference with paged attention
- **TensorRT-LLM**: NVIDIA's optimized inference framework
- **DeepSpeed-FastGen**: Inference optimizations
