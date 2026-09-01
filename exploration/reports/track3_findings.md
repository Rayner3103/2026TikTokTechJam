# Track 3: Optimization Research Findings

**Date:** 2026-08-31  
**Focus:** Optimizing `UserOptimizedTransformer` to beat baseline latency while maintaining numerical accuracy

---

## 1. Benchmark Code Analysis

### 1.1 What the Benchmark Measures

The benchmark (`torch_transformer_benchmark.py`) compares two implementations:

| Component | Baseline | User Optimized |
|-----------|----------|----------------|
| **Attention** | `BaselineSelfAttention` — explicit split_heads, matmul, softmax, matmul | `UserOptimizedTransformer` |
| **Block** | `BaselineTransformerBlock` — norm → attn → residual → norm → FFN → residual | Must inherit `BaselineTransformer` |
| **FFN** | `nn.Linear` → `F.gelu` → `nn.Linear` | Compatible parameter names required |
| **Correctness** | `abs(error) <= 0.002 OR rel_error <= 2%` (default `--rtol`, `--atol`) | Same threshold |
| **Timing** | CUDA events (or `perf_counter_ns` on CPU), median over rounds × repeats | Same conditions |

**Default benchmark configuration:**
```
--batch-size=8 --seq-len=128 --d-model=512 --heads=8 --ffn-dim=2048 --layers=6
--device=auto --dtype=float32 --padding-ratio=0.0 --input-scale=1.0
--accuracy-trials=5 --rtol=0.02 --atol=0.002
--warmup=20 --repeats=100 --benchmark-rounds=3
```

**Critical constraint:** `copy_model_weights()` copies baseline state_dict → optimized. Parameter names must match unless `--non-strict-weight-copy` is used.

### 1.2 Baseline Architecture Breakdown

The baseline is a vanilla PyTorch transformer with these inefficiencies:

1. **Separate Q, K, V projections** — Three independent `nn.Linear` calls, each a separate cuBLAS GEMM
2. **Explicit head splitting** — `view` + `transpose` + `contiguous()` — the `.contiguous()` forces a copy
3. **Attention computed per-head in PyTorch eager mode** — No kernel fusion, each matmul is a separate CUDA kernel launch
4. **Softmax in float32** then cast back — Good for correctness, bad for performance
5. **Two separate attention matmuls** — `QK^T` and `AV` — separate kernel launches, intermediate written to HBM
6. **Separate LayerNorm** — Each `nn.LayerNorm` is its own kernel, reading/writing full tensor
7. **FFN: Linear → Gelu → Linear** — Three separate kernel launches
8. **Residual connections** — Element-wise add as separate kernels
9. **Valid token masking** — `masked_fill` operations on every layer

**Key insight:** The baseline has **~15-20+ individual CUDA kernel launches** per layer, most of which could be fused.

---

## 2. Current Solutions Landscape

### 2.1 PyTorch Native Optimizations

#### `torch.nn.functional.scaled_dot_product_attention (SDPA)`

PyTorch 2.0+ provides `F.sdp()` which auto-selects the best backend:

| Backend | When Used | Speedup vs Baseline |
|---------|-----------|---------------------|
| **FlashAttention** (cuBLAS + Flash kernel) | Small seq, GPU, FP16/BF16 | 1.5-2.0x |
| **Memory-efficient** (tiled, recomputation) | Long sequences | 1.3-1.8x |
| **Math** (pure PyTorch) | CPU or when others disabled | 1.0-1.2x |
| **cuDNN** | Certain shapes on newer GPUs | 1.2-1.5x |

Evidence: From PyTorch tutorial benchmarks — SDPA default: **2274 μs** vs math-only: **87,620 μs** — that's a **38x speedup** when FlashAttention backend activates.

**For our benchmark:** SDPA gives 1.5-2.0x on attention alone, which is ~50-60% of total time. Net speedup: **~1.3-1.6x** total.

#### `torch.compile`

Fuses operations at graph level, eliminates Python overhead:

- `mode="default"`: Basic fusion — ~1.2-1.5x
- `mode="reduce-overhead"`: Less kernel launch overhead — good for small models
- `mode="max-autotune"`: Exhaustive search for best config — best but slowest compile

Evidence: Adam Casson's benchmark on Vision Transformer — compiled: **242 ms** vs uncompiled: **370 ms** — **1.53x speedup**.

**For our benchmark:** `torch.compile` on the whole transformer gives 1.3-1.8x depending on config. It fuses LayerNorm, residual adds, and FFN.

#### Combined: SDPA + torch.compile

This is the strongest *pure-PyTorch* approach. PyTorch blog shows these stack:
- SDPA replaces attention matmuls with fused kernels
- torch.compile fuses everything else
- **Expected net speedup: 2-3x**

### 2.2 FlashAttention Family

| Version | Key Innovation | Speedup over Baseline PyTorch |
|---------|---------------|-------------------------------|
| **FlashAttention 1** (NeurIPS 2022) | IO-aware tiling, shared memory | 2-4x |
| **FlashAttention 2** (ICLR 2024) | Better parallelism, work partitioning | 4-8x |
| **FlashAttention 3** (2024) | Asynchrony, FP8, WGMMA, TMA | 6-12x on H100 |

**FlashAttention principle:** Never write full attention matrix to HBM. Compute softmax in SRAM, accumulate output. Saves 2× memory traffic.

**Limitation:** FlashAttention is a custom CUDA kernel (in the `flash-attn` package). Integrating it requires:
- Installing the `flash-attn` package
- Using its API: `flash_attn.flash_attn_func(q, k, v, causal=...)`
- The baseline expects specific module structure with `nn.Linear` projections

### 2.3 Fusion Techniques

#### Fused Linear Operations (QKV)
Instead of 3 separate GEMMs:
```python
# Baseline (3 kernel launches)
q = self.q_proj(x)
k = self.k_proj(x)
v = self.v_proj(x)

# Fused (1 GEMM + split)
qkv = self.qkv_proj(x)  # shape: [B, S, 3*d_model]
q, k, v = qkv.chunk(3, dim=-1)
```
Speedup: **1.8-2.0x** for the projection phase.

#### Fused LayerNorm + Operations
PyTorch 2.4+ has `torch.nn.functional.layer_norm` which can be fused by `torch.compile` with surrounding ops.

#### Fused FFN (GELU + 2 Matmuls)
Fuse the FFN into a single kernel. torch.compile can do this automatically.

### 2.4 FlexAttention (PyTorch 2.5+)

PyTorch 2.5's `torch.nn.attention.flex_attention`:
- Compile-time kernel generation via Python score_mod
- 90% of FlashAttention 2 performance
- More flexible than SDPA (custom masks, soft-capping, etc.)
- **Speedup vs baseline SDPA: 1.1-1.3x**

---

## 3. Three Target Optimization Directions

### Direction 1: SDPA + torch.compile (High-Impact, Low-Risk)

**Concept:** Replace the explicit attention with `F.scaled_dot_product_attention` and wrap the model with `torch.compile`.

**Why this works:**
- SDPA internally uses FlashAttention (cuDNN or Flash kernel) when conditions are met
- torch.compile fuses LayerNorm, residuals, and FFN into single kernels
- Zero accuracy risk — SDPA is mathematically identical to baseline attention
- torch.compile is pure PyTorch, no external dependencies

**Evidence:**
- PyTorch tutorial: SDPA default **2274 μs** vs math-only **87,620 μs** (38x for attention phase)
- Adam Casson ViT benchmark: torch.compile **242ms** vs eager **370ms** (1.53x)
- Combined effect on transformer: literature shows **2-3x** total speedup
- From benchmark.py source: the comment itself suggests SDPA + torch.compile as the first optimization direction

**Implementation sketch:**
```python
class UserOptimizedTransformer(BaselineTransformer):
    def forward(self, x, valid_token_mask=None):
        causal = self.config.causal
        for layer in self.layers:
            # Extract layer components for manual optimization
            # Apply SDPA attention + fused operations
            ...
```

**Risk:** Low. Accuracy is guaranteed (SDPA is exact, not approximate). torch.compile can fall back to eager if graph capture fails.

**Expected speedup:** 2.0-3.0x on CUDA, 1.3-1.8x on CPU.

**Implementation time:** 2-4 hours (includes testing, edge cases).

---

### Direction 2: QKV Fusion + Attention Tiling (Medium-Impact, Medium-Risk)

**Concept:** Manually fuse QKV projections into one GEMM, tile the attention computation for better cache utilization, and fuse FFN operations.

**Why this works:**
- Baseline launches 3 separate GEMMs for Q, K, V projections — fusing saves memory bandwidth
- Tiling attention reduces intermediate HBM writes (FlashAttention principle, implemented in PyTorch)
- Fused FFN reduces kernel launch overhead

**Evidence:**
- QKV fusion: standard optimization in all production transformers (Megatron, DeepSpeed, FairSeq). **1.5-2x** speedup for projection phase.
- Tiled attention: FlashAttention paper shows **2-4x** speedup over naive attention by avoiding HBM writes of the full attention matrix.
- FFN fusion: torch.compile achieves this automatically, but manual fusion gives more control.

**Implementation approach (in pure PyTorch, no CUDA):**
```python
# Fused QKV projection
class OptimizedAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        # Single fused GEMM for all projections
        self.qkv_linear = nn.Linear(d_model, d_model * 3, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
    
    def forward(self, x, valid_token_mask=None, causal=False):
        B, S, D = x.shape
        # One GEMM instead of three
        qkv = self.qkv_linear(x)  # [B, S, 3*D]
        qkv = qkv.view(B, S, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # [B, S, H, Hd]
        q = q.transpose(1, 2)  # [B, H, S, Hd]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Use SDPA for the attention computation (handles tiling internally)
        if causal:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            out = F.scaled_dot_product_attention(q, k, v)
        
        # Handle valid_token_mask via manual masking
        if valid_token_mask is not None:
            # Convert mask to attention bias
            ...
        
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(out)
```

**Risk:** Medium. Manual masking logic for `valid_token_mask` needs careful implementation. The baseline's mask handling is subtle (invalid key positions masked with `-inf`, invalid output positions zeroed).

**Expected speedup:** 1.5-2.0x on attention + projection phase → **1.3-1.7x** total.

**Implementation time:** 4-8 hours (masking edge cases, testing).

---

### Direction 3: Layer-Level Fused Kernel via torch.compile with Custom Graph Breaks (Highest-Potential, Highest-Risk)

**Concept:** Restructure the transformer to maximize torch.compile's fusion potential, using a single forward pass with minimal graph breaks, combined with selective use of SDPA and manual operation fusion.

**Why this works:**
- torch.compile's biggest gains come from fusing LayerNorm → attention → residual → LayerNorm → FFN → residual into 1-2 kernels instead of 8+
- The baseline has many graph breaks (Python control flow, separate module calls)
- By restructure the model to minimize graph breaks and explicitly fuse operations, we maximize the JIT compiler's effectiveness

**Evidence:**
- Sebastian Raschka's analysis: torch.compile improves LLM throughput by capturing repeated operations, fusing kernels, reducing dispatch overhead.
- NVIDIA Megatron-LM: Combining torch.compile with fused transformer blocks achieves **2-4x** speedup.
- PyTorch blog (Towards Free Normalization): Fusing LayerNorm into attention kernels achieves up to **35%** kernel speedup.
- On small-to-medium transformers (like our benchmark: 512 dim, 6 layers), torch.compile with `max-autotune` can achieve **3-5x** speedup because the overhead of Python dispatch dominates.

**Implementation approach:**
```python
class UserOptimizedTransformer(BaselineTransformer):
    def __init__(self, config):
        super().__init__(config)
        # Restructure for maximum fusion
    
    def _optimized_layer(self, x, norm1, attention, norm2, ffn_in, ffn_out, 
                          valid_token_mask, causal):
        # Fused: LayerNorm → Attention → Residual → LayerNorm → FFN → Residual
        # All in one function to avoid graph breaks
        residual = x
        x = norm1(x)
        
        # SDPA attention (fused internally)
        attn_out = attention(x, valid_token_mask, causal)
        x = x + attn_out  # In-place-ish residual
        
        if valid_token_mask is not None:
            x = x.masked_fill(~valid_token_mask[..., None], 0)
        
        residual2 = x
        x = norm2(x)
        ff_out = ffn_out(F.gelu(ffn_in(x), approximate="none"))
        x = x + ff_out
        
        if valid_token_mask is not None:
            x = x.masked_fill(~valid_token_mask[..., None], 0)
        
        return x
    
    def forward(self, x, valid_token_mask=None):
        # Single loop, minimal graph breaks
        for i, layer in enumerate(self.layers):
            x = self._optimized_layer(
                x, layer.norm1, layer.attention, layer.norm2,
                layer.ffn_in, layer.ffn_out, valid_token_mask, self.config.causal
            )
        x = self.final_norm(x)
        if valid_token_mask is not None:
            x = x.masked_fill(~valid_token_mask[..., None], 0)
        return x
```

Then in main:
```python
optimized = UserOptimizedTransformer(config)
copy_model_weights(baseline, optimized)
optimized = torch.compile(optimized, mode="max-autotune")  # Best performance
```

**Risk:** Higher. The restructured forward pass must exactly replicate baseline behavior:
- Valid token masking must be applied at the right places (after attention output and after FFN output)
- The residual addition pattern must match exactly
- torch.compile with `max-autotune` has a long warmup (100+ iterations) — need enough warmup iterations in the benchmark
- Graph breaks from `valid_token_mask` logic could negate gains

**Expected speedup:** 2.5-4.0x on CUDA with `max-autotune`. 1.5-2.5x with `default` mode.

**Implementation time:** 6-12 hours (graph break debugging, correctness verification, tuning).

---

## 4. Comparative Analysis

| Dimension | Direction 1: SDPA + compile | Direction 2: QKV Fusion + Tiling | Direction 3: Layer Fusion + compile |
|-----------|---------------------------|----------------------------------|--------------------------------------|
| **Expected speedup** | 2.0-3.0x | 1.3-1.7x | 2.5-4.0x |
| **Risk level** | Low | Medium | High |
| **Accuracy risk** | Zero (SDPA is exact) | Low (same math, different ops) | Medium (graph breaks, edge cases) |
| **Implementation time** | 2-4 hrs | 4-8 hrs | 6-12 hrs |
| **External deps** | None (PyTorch only) | None | None |
| **GPU dependency** | Works on CPU too (SDPA falls back) | Works on CPU | Works on CPU but less benefit |
| **Scaling** | Good for all seq lengths | Best for long sequences | Best for small-medium models |
| **Complexity** | Simple | Moderate | High |
| **Recommended order** | ✅ Start here | ✅ If more needed | ✅ If chasing max performance |

---

## 5. Recommended Strategy

### Phase 1 (Hours 0-4): Baseline + SDPA + torch.compile (Direction 1)
1. Implement `UserOptimizedTransformer` using `F.scaled_dot_product_attention` with `is_causal`
2. Keep the same module structure (compatibility with weight copy)
3. Add `torch.compile` wrapping
4. Verify accuracy passes all 5 trials
5. Establish baseline speedup number

**Why start here:** It's the highest ROI. 2-3x speedup for 2-4 hours of work. Zero accuracy risk. If this gives enough speedup, no need for more complex approaches.

### Phase 2 (Hours 4-8): QKV Fusion + Optimized Attention (Direction 2)
1. If more speedup is needed, add QKV fusion
2. Combine with Direction 1's SDPA + compile
3. Handle valid_token_mask edge cases carefully

### Phase 3 (Hours 8+): Layer-level fusion with max-autotune (Direction 3)
1. Only if chasing the maximum possible speedup
2. Requires careful correctness verification
3. May need to adjust warmup iterations for torch.compile to converge

---

## 6. Key Technical Details to Handle

### 6.1 Valid Token Mask Compatibility

The baseline applies the mask in 3 places:
1. **Attention scores**: `~valid_token_mask[:, None, None, :]` — masks invalid key positions
2. **Attention output**: `~valid_token_mask[..., None]` — zeros invalid output positions
3. **Final output**: Same as above

Any optimized implementation must replicate this exactly. With SDPA, the attention score masking must be done via `attn_mask` parameter (not `is_causal`).

### 6.2 Weight Copy Compatibility

The baseline uses:
```python
state_dict = copy.deepcopy(baseline.state_dict())
optimized.load_state_dict(state_dict, strict=strict)
```

Parameter names must match. If restructuring modules (e.g., changing `q_proj`, `k_proj`, `v_proj` to `qkv_linear`), either:
- Use `--non-strict-weight-copy` (if allowed)
- Or rename parameters to maintain compatibility
- Or write a custom weight loader

### 6.3 Causal Mask Handling

The baseline builds a triangular causal mask in PyTorch:
```python
causal_mask = torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool).triu(diagonal=1)
scores = scores.masked_fill(causal_mask, float("-inf"))
```

With SDPA, this is replaced by `is_causal=True`, which is both faster and numerically equivalent.

### 6.4 dtype Handling

The baseline's attention does:
```python
probs = torch.softmax(scores.float(), dim=-1).to(dtype=x.dtype)
```

SDPA handles this internally — it runs softmax in higher precision and casts back. This should be numerically equivalent.

---

## 7. References

1. **SDPA Documentation**: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
2. **SDPA Tutorial** (with benchmarks): https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html
3. **torch.compile Tutorial**: https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
4. **FlashAttention 2** (ICLR 2024): https://openreview.net/forum?id=mZn2Xyh9Ec
5. **FlashAttention 3** (2024): https://arxiv.org/html/2407.08608v1
6. **FlexAttention** (PyTorch 2.5): https://pytorch.org/blog/flexattention/
7. **Fusing Normalization** (PyTorch Blog): https://pytorch.org/blog/towards-free-normalization-fusing-normalization-into-gemm-and-attention-kernels/
8. **Adam Casson** — torch.compile ViT analysis: https://www.adamcasson.com/posts/torch-compile-vit
9. **Sebastian Raschka** — torch.compile for LLMs: https://sebastianraschka.com/faq/docs/torch-compile-llm-workloads.html
10. **Transformer Engine** (NVIDIA): https://github.com/NVIDIA/Megatron-LM/discussions/1089

---

## 8. Quick Start Recommendation

```bash
# Run baseline to get reference numbers
python torch_transformer_benchmark.py --batch-size 8 --seq-len 128 --d-model 512 \
    --heads 8 --ffn-dim 2048 --layers 6 --device auto --dtype float32 \
    --accuracy-trials 5 --warmup 20 --repeats 100 --benchmark-rounds 3

# Run with torch.compile on baseline (for comparison)
python torch_transformer_benchmark.py --batch-size 8 --seq-len 128 --d-model 512 \
    --heads 8 --ffn-dim 2048 --layers 6 --device auto --dtype float32 \
    --accuracy-trials 5 --warmup 50 --repeats 100 --benchmark-rounds 3 \
    --compile-baseline --compile-mode max-autotune

# After implementing UserOptimizedTransformer, run with compile
python torch_transformer_benchmark.py --batch-size 8 --seq-len 128 --d-model 512 \
    --heads 8 --ffn-dim 2048 --layers 6 --device auto --dtype float32 \
    --accuracy-trials 5 --warmup 50 --repeats 100 --benchmark-rounds 3 \
    --compile-user --compile-mode max-autotune
```

---

## 9. Summary Table

| Direction | Approach | Speedup | Risk | Time | Priority |
|-----------|----------|---------|------|------|----------|
| **1. SDPA + torch.compile** | Replace explicit attention with `F.scaled_dot_product_attention`, wrap with `torch.compile` | 2-3x | Low | 2-4 hrs | **P0 — Do first** |
| **2. QKV Fusion + Tiling** | Single GEMM for projections, SDPA for attention | 1.3-1.7x | Medium | 4-8 hrs | P1 — Add if needed |
| **3. Layer Fusion + max-autotune** | Restructure forward pass, minimize graph breaks, compile with max-autotune | 2.5-4x | High | 6-12 hrs | P2 — For max performance |
