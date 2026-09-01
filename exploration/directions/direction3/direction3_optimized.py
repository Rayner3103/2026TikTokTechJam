#!/usr/bin/env python3
"""
Direction 3: Layer-Level Fused Kernel + torch.compile max-autotune

Approach: Restructure the transformer to maximize torch.compile's fusion potential
by minimizing graph breaks and fusing operations at the layer level.

Key optimizations:
1. Custom fused forward pass — minimizes graph breaks
2. SDPA attention (fused internally by torch.compile)
3. torch.compile(mode="max-autotune") — exhaustive kernel search
4. Manual residual handling to avoid graph breaks

Expected speedup: 2.5-4x on CUDA, 1.5-2.5x on CPU
Risk: High (graph break debugging, edge cases, torch.compile warmup)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FusedAttentionLayer(nn.Module):
    """Attention sub-layer with SDPA, minimal graph breaks."""
    
    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.norm1 = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

    def forward(self, x: torch.Tensor, valid_token_mask: torch.Tensor | None, causal: bool) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        
        # LayerNorm
        normed = self.norm1(x)
        
        # Q, K, V projections
        q = self.q_proj(normed)
        k = self.k_proj(normed)
        v = self.v_proj(normed)
        
        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # SDPA attention with proper bias handling
        if causal and valid_token_mask is None:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.scale)
        elif causal and valid_token_mask is not None:
            invalid_keys = ~valid_token_mask[:, None, None, :]  # [B, 1, 1, S]
            key_bias = invalid_keys.masked_fill(invalid_keys.bool(), float('-inf'))  # [B, 1, 1, S]
            causal_bias = torch.ones((seq_len, seq_len), device=x.device, dtype=x.dtype).triu(1) * -1e9
            causal_bias = causal_bias[None, None, :, :]  # [1, 1, L, S]
            key_bias = key_bias.expand(-1, -1, seq_len, -1)  # [B, 1, L, S]
            attn_bias = causal_bias + key_bias
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, scale=self.scale)
        elif valid_token_mask is not None:
            invalid_keys = ~valid_token_mask[:, None, None, :]  # [B, 1, 1, S]
            attn_bias = invalid_keys.masked_fill(invalid_keys.bool(), float('-inf'))  # [B, 1, 1, S]
            attn_bias = attn_bias.expand(-1, -1, seq_len, -1)  # [B, 1, L, S]
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, scale=self.scale)
        else:
            out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        
        # Merge heads
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        attn_out = self.out_proj(out)
        
        # Residual
        x = x + attn_out
        
        if valid_token_mask is not None:
            x = x.masked_fill(~valid_token_mask[..., None], 0)
        
        return x


class FusedFFNLayer(nn.Module):
    """FFN sub-layer."""
    
    def __init__(self, d_model: int, ffn_dim: int) -> None:
        super().__init__()
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn_in = nn.Linear(d_model, ffn_dim)
        self.ffn_out = nn.Linear(ffn_dim, d_model)

    def forward(self, x: torch.Tensor, valid_token_mask: torch.Tensor | None) -> torch.Tensor:
        residual = x
        x = self.norm2(x)
        x = self.ffn_out(F.gelu(self.ffn_in(x), approximate="none"))
        x = residual + x
        
        if valid_token_mask is not None:
            x = x.masked_fill(~valid_token_mask[..., None], 0)
        return x


class FusedTransformerBlock(nn.Module):
    """Single transformer block combining attention + FFN."""
    
    def __init__(self, d_model: int, num_heads: int, ffn_dim: int) -> None:
        super().__init__()
        self.attn_layer = FusedAttentionLayer(d_model, num_heads)
        self.ffn_layer = FusedFFNLayer(d_model, ffn_dim)

    def forward(self, x: torch.Tensor, valid_token_mask: torch.Tensor | None, causal: bool) -> torch.Tensor:
        x = self.attn_layer(x, valid_token_mask, causal)
        x = self.ffn_layer(x, valid_token_mask)
        return x


class OptimizedTransformer(nn.Module):
    """
    Optimized transformer for maximum torch.compile fusion.
    
    Architecture matches baseline for weight compatibility.
    torch.compile will fuse LayerNorm + attention + residual + FFN into minimal kernels.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [
                FusedTransformerBlock(
                    config.d_model, config.num_heads, config.ffn_dim
                )
                for _ in range(config.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, valid_token_mask, self.config.causal)
        x = self.final_norm(x)
        if valid_token_mask is not None:
            x = x.masked_fill(~valid_token_mask[..., None], 0)
        return x


if __name__ == "__main__":
    print("Direction 3: Layer-Level Fused Kernel + torch.compile max-autotune")
    print("Implementation: direction3_optimized.py")
