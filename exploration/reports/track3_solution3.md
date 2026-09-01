# Solution 3: Causal Linear Attention with O(N) Complexity

**Importance: ███████░░░ (7/10) — Innovative and research-worthy**
**Ease of Implementation: ██████░░░░ (6/10) — Medium complexity**

## Problem Solved

Addresses the **core challenge of long-sequence transformer inference** — reducing attention complexity from O(N²) to O(N) for long sequences, which is increasingly important for production LLM deployments.

## Concept

Implement a **linear attention kernel** that computes attention in O(N) time and space instead of O(N²). This uses kernel-based feature maps (e.g., ELU, ReLU) to reformulate attention as:

```
Attention(Q, K, V) = φ(Q) (φ(K)^T V) / φ(Q) K^T
```

where φ is a feature map (e.g., ELU + 1). This enables processing very long sequences efficiently.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│              Linear Attention Kernel                     │
│                                                          │
│  ┌─────────────┐    ┌─────────────┐                    │
│  │   Feature   │    │   Prefix    │                    │
│  │   Map φ(·)  │    │   Sum       │                    │
│  │             │    │             │                    │
│  │  Q' = φ(Q)  │    │  K'V = K'^T V │                  │
│  │  K' = φ(K)  │    │  (O(N) space)│                   │
│  └──────┬──────┘    └──────┬──────┘                    │
│         │                   │                            │
│         └────────┬──────────┘                            │
│                  ▼                                        │
│         ┌────────────────────┐                           │
│         │  Linear Attention   │                           │
│         │                    │                           │
│         │  Output = Q' K'V /  │                           │
│         │  (Q' K'^T)          │                           │
│         └────────────────────┘                           │
│                                                          │
│  ┌─────────────────────────────────┐                   │
│  │  GPU Optimizations              │                   │
│  │                                  │                   │
│  │  • Fused feature map            │                   │
│  │  • Sequential prefix sum        │                   │
│  │  • Register tiling              │                   │
│  │  • Streaming V computation      │                   │
│  └─────────────────────────────────┘                   │
└─────────────────────────────────────────────────────────┘
```

## Technical Implementation

### 1. Feature Map Implementation
```python
import triton
import triton.language as tl

@triton.jit
def elu_feature_map(x):
    """ELU + 1 feature map: φ(x) = elu(x) + 1"""
    return tl.maximum(0, x) + x + 1

@triton.jit
def softmax_feature_map(x):
    """Softmax feature map (approximate)"""
    x_max = tl.max(x, axis=-1)
    x_shifted = x - x_max[..., None]
    return tl.exp(x_shifted)

@triton.jit
def relu_feature_map(x):
    """ReLU feature map: φ(x) = max(0, x)"""
    return tl.maximum(0, x)
```

### 2. Linear Attention Kernel
```python
@triton.jit
def linear_attention_kernel(
    Q, K, V,
    Output,
    stride_q_batch, stride_q_seq, stride_q_head,
    stride_k_batch, stride_k_seq, stride_k_head,
    stride_v_batch, stride_v_seq, stride_v_head,
    stride_out_batch, stride_out_seq, stride_out_head,
    Batch, SeqLen, NumHeads, HeadDim,
    FEATURE_MAP: tl.constexpr,  # 'elu', 'relu', 'softmax'
    BLOCK_SIZE: tl.constexpr,
):
    """
    Linear attention: O(N) instead of O(N²)
    Attention(Q,K,V) = φ(Q) (Σ φ(k_i) v_i^T) / Σ φ(q_i) k_i^T
    """
    pid = tl.program_id(0)
    
    # Iterate over sequence positions
    for start_pos in range(0, SeqLen, BLOCK_SIZE):
        end_pos = min(start_pos + BLOCK_SIZE, SeqLen)
        block_len = end_pos - start_pos
        
        # Load Q, K, V tiles
        q_ptrs = Q + pid * SeqLen * HeadDim + start_pos * HeadDim + tl.arange(0, HeadDim)
        k_ptrs = K + pid * SeqLen * HeadDim + start_pos * HeadDim + tl.arange(0, HeadDim)
        v_ptrs = V + pid * SeqLen * HeadDim + start_pos * HeadDim + tl.arange(0, HeadDim)
        
        q = tl.load(q_ptrs)
        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)
        
        # Apply feature map
        if FEATURE_MAP == 'elu':
            q_feat = elu_feature_map(q)
            k_feat = elu_feature_map(k)
        elif FEATURE_MAP == 'relu':
            q_feat = relu_feature_map(q)
            k_feat = relu_feature_map(k)
        else:
            q_feat = softmax_feature_map(q)
            k_feat = softmax_feature_map(k)
        
        # Update prefix sums (running totals)
        # This is the key O(N) trick: accumulate instead of computing full matrix
        pass  # Simplified for hackathon scope
    
    # Compute final output
    # Output = Q_feat @ (K_feat^T @ V) / (Q_feat @ K_feat^T)
    pass
```

### 3. Prefix Sum for Linear Attention
```python
@triton.jit
def prefix_sum_kernel(
    input, output, stride, length,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Parallel prefix sum (scan) for computing attention normalization
    """
    pid = tl.program_id(0)
    
    # Load block
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < length
    data = tl.load(input + pid * BLOCK_SIZE + offsets, mask=mask)
    
    # Sequential scan within block
    for i in range(1, BLOCK_SIZE):
        data[i] += data[i-1]
    
    # Store result
    tl.store(output + pid * BLOCK_SIZE + offsets, data, mask=mask)
```

### 4. Multi-Head Linear Attention
```python
class LinearAttention(torch.nn.Module):
    def __init__(self, dim, num_heads=8, head_dim=None, feature_map='elu'):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim or dim // num_heads
        self.feature_map = feature_map
        self.qkv_proj = torch.nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = torch.nn.Linear(dim, dim)
    
    def forward(self, x):
        B, L, D = x.shape
        
        # Project to Q, K, V
        qkv = self.qkv_proj(x)  # (B, L, 3D)
        q, k, v = qkv.chunk(3, dim=-1)  # (B, L, D) each
        
        # Reshape for multi-head
        q = q.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, L, D/H)
        k = k.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute linear attention per head
        output = []
        for h in range(self.num_heads):
            q_h = q[:, h]  # (B, L, D/H)
            k_h = k[:, h]
            v_h = v[:, h]
            
            # Apply feature map
            if self.feature_map == 'elu':
                q_feat = torch.nn.functional.elu(q_h) + 1
                k_feat = torch.nn.functional.elu(k_h) + 1
            elif self.feature_map == 'relu':
                q_feat = torch.nn.functional.relu(q_h)
                k_feat = torch.nn.functional.relu(k_h)
            else:
                q_feat = torch.nn.functional.softmax(q_h, dim=-1)
                k_feat = torch.nn.functional.softmax(k_h, dim=-1)
            
            # Linear attention: O(N)
            # KV_sum = Σ k_feat_i ⊗ v_i^T  (outer product, accumulated)
            # QK_sum = Σ q_feat_i ⊗ k_i  (dot product, accumulated)
            
            KV_sum = torch.zeros(B, self.head_dim, self.head_dim, device=x.device)
            QK_sum = torch.zeros(B, self.head_dim, device=x.device)
            
            for i in range(L):
                KV_sum += torch.einsum('d,de->de', k_feat[:, i], v_h[:, i])
                QK_sum += torch.einsum('d,d->', q_feat[:, i], k_feat[:, i])
            
            # Output = (Q_feat @ KV_sum) / (QK_sum + eps)
            out_h = torch.einsum('bd,de->be', q_feat, KV_sum)
            out_h = out_h / (QK_sum.unsqueeze(-1) + 1e-6)
            
            output.append(out_h)
        
        out = torch.cat(output, dim=-1).transpose(1, 2).reshape(B, L, D)
        return self.out_proj(out)
```

### 5. GPU-Fused Prefix Attention
```python
@triton.jit
def fused_prefix_attention_kernel(
    Q, K, V, Output,
    Batch, SeqLen, NumHeads, HeadDim,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    """
    Fully fused prefix attention on GPU
    - Loads K,V tiles and accumulates KV product in shared memory
    - Processes Q tiles and computes attention with prefix sums
    - Single kernel launch for entire attention
    """
    pass  # Simplified for hackathon
```

## Advantages of Linear Attention

| Aspect | Standard Attention | Linear Attention |
|--------|-------------------|-----------------|
| Complexity | O(N²) time, O(N²) space | O(N) time, O(N²) space |
| Long sequences | Very slow for N > 2048 | Scales linearly |
| Parallelism | Excellent | Good (less data dependency) |
| Accuracy | Exact | Approximate (depends on feature map) |
| Memory | HBM bound | Compute bound |

## Trade-offs & Considerations

1. **Approximation Error**: Linear attention is approximate; accuracy depends on feature map choice
2. **KV Product Memory**: O(N²) space for KV product (but only one, not N² matrix)
3. **Sequential Dependency**: Prefix sum creates data dependency; limits parallelism
4. **Feature Map Choice**: ELU typically best; ReLU simplest; Softmax most accurate but slowest

## UI Integration

- **Complexity Comparison**: Interactive visualization showing O(N²) vs O(N) scaling
- **Feature Map Selector**: Switch between ELU, ReLU, Softmax and compare accuracy
- **Sequence Length Slider**: See how performance scales with sequence length
- **Accuracy vs Speed Trade-off**: Pareto chart of approximation quality vs speedup

## Justification

- **Medium-High Impact**: O(N) attention is cutting-edge research; demonstrates innovation
- **Medium Difficulty**: Simpler kernel than FlashAttention (no tiling complexity)
- **Hackathon Feasible**: PyTorch implementation first, then GPU-fuse the critical path
- **Evaluatable**: Clear scaling benefits for long sequences; accuracy metrics

## References

- Linear Attention: https://arxiv.org/abs/2006.16236
- Performer (Fast Attention): https://arxiv.org/abs/2009.14794
- RetNet: https://arxiv.org/abs/2307.08621
- FlashLinearAttention: https://github.com/HazyResearch/flash-attention/tree/main/transformers
