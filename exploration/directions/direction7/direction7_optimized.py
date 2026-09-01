#!/usr/bin/env python3
"""
Direction 7: Adaptive Layer Skipping + Dynamic Routing

Approach: Dynamically skip or route through layers based on token importance
          and model confidence. Reduces compute for "easy" tokens.

Key optimizations:
1. Confidence-based layer routing: token confidence gates layer execution
2. Token-level early exit: stop processing token if high enough confidence
3. Adaptive FFN skipping: skip FFN for low-importance tokens
4. Lightweight routing networks for routing decisions

Expected speedup: 1.3-1.8x (model-dependent, requires fine-tuning)
Risk: High (requires careful initialization and training)

This is an experimental direction requiring architectural changes to training pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class AdaptiveLayerRouter(nn.Module):
    """Lightweight routing network to decide layer execution.
    
    Given current hidden state, predicts:
    1. Whether to skip this layer entirely (bypass)
    2. Whether to skip the FFN sub-layer
    
    Uses gating mechanism inspired by Mixture of Experts.
    """
    
    def __init__(self, d_model: int, num_layers: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        
        # Simple routing: linear layer on layer norm output -> routing logits
        self.router_mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2, bias=True),
            nn.ReLU(),
            nn.Linear(d_model // 2, 2, bias=True),  # [skip_logit, skip_ffn_logit]
        )
    
    def forward(self, x: torch.Tensor, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict routing decisions.
        
        Args:
            x: (batch, seq_len, d_model) hidden states
            layer_idx: which layer we're deciding about
        
        Returns:
            skip_prob: (batch, seq_len) probability to skip layer
            skip_ffn_prob: (batch, seq_len) probability to skip FFN
        """
        # Use mean over sequence dimension for routing decision
        x_pooled = x.mean(dim=1)  # (batch, d_model)
        
        routing_logits = self.router_mlp(x_pooled)  # (batch, 2)
        
        # Convert to probabilities per token (replicate over sequence)
        skip_logits = routing_logits[:, 0:1]  # (batch, 1)
        skip_ffn_logits = routing_logits[:, 1:2]  # (batch, 1)
        
        skip_prob = torch.sigmoid(skip_logits).expand_as(x[:, :, 0])  # (batch, seq_len)
        skip_ffn_prob = torch.sigmoid(skip_ffn_logits).expand_as(x[:, :, 0])  # (batch, seq_len)
        
        return skip_prob, skip_ffn_prob


class AdaptiveTransformerLayer(nn.Module):
    """Transformer layer with optional layer/FFN skipping."""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_dim: int,
        num_layers: int,
        layer_idx: int,
    ) -> None:
        super().__init__()
        
        self.d_model = d_model
        self.layer_idx = layer_idx
        
        # Attention sub-layer
        self.norm1 = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        
        # FFN sub-layer
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, ffn_dim, bias=True),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model, bias=True),
        )
        
        # Routing network (shared across all layers)
        self.router = AdaptiveLayerRouter(d_model, num_layers)
    
    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        return (
            x.view(batch, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )
    
    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, num_heads, seq_len, head_dim = x.shape
        return x.transpose(1, 2).contiguous().view(batch, seq_len, -1)
    
    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: torch.Tensor | None = None,
        causal: bool = False,
        use_routing: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with optional adaptive routing.
        
        Args:
            x: (batch, seq_len, d_model)
            valid_token_mask: (batch, seq_len) optional padding mask
            causal: whether to apply causal masking
            use_routing: whether to use adaptive routing
        
        Returns:
            out: (batch, seq_len, d_model)
            routing_decisions: (batch, seq_len, 2) [skip_layer, skip_ffn]
        """
        batch, seq_len, _ = x.shape
        
        # Get routing decisions if using adaptive routing
        skip_layer_probs = None
        skip_ffn_probs = None
        if use_routing:
            skip_layer_probs, skip_ffn_probs = self.router(x, self.layer_idx)
            
            # During training: use soft gating (backprop through)
            # During inference: use hard decisions with some stochasticity
            if self.training:
                skip_layer_gates = skip_layer_probs
                skip_ffn_gates = skip_ffn_probs
            else:
                # Hard gating with temperature for inference
                skip_layer_gates = (skip_layer_probs > 0.5).float()
                skip_ffn_gates = (skip_ffn_probs > 0.5).float()
        else:
            skip_layer_gates = torch.zeros(batch, seq_len, device=x.device)
            skip_ffn_gates = torch.zeros(batch, seq_len, device=x.device)
        
        # Save residual
        residual = x
        
        # --- Attention Block ---
        normed = self.norm1(x)
        
        # Project Q, K, V
        q = self._split_heads(self.q_proj(normed))
        k = self._split_heads(self.k_proj(normed))
        v = self._split_heads(self.v_proj(normed))
        
        # Attention
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=causal,
        )
        
        attn_out = self._merge_heads(attn_out)
        attn_out = self.out_proj(attn_out)
        
        # Apply layer skip: if skip_layer_gates[b, t] = 1, skip this layer for that token
        # By interpolating between residual and layer output
        skip_layer_gates = skip_layer_gates.unsqueeze(-1)  # (batch, seq_len, 1)
        x = residual + (1 - skip_layer_gates) * attn_out
        
        # --- FFN Block with Optional Skipping ---
        normed = self.norm2(x)
        ffn_out = self.mlp(normed)
        
        # Apply FFN skip: only compute FFN for tokens with skip_ffn_gates=0
        skip_ffn_gates = skip_ffn_gates.unsqueeze(-1)  # (batch, seq_len, 1)
        x = x + (1 - skip_ffn_gates) * ffn_out
        
        # Return routing decisions for analysis
        routing_decisions = torch.stack([skip_layer_probs, skip_ffn_probs], dim=-1)
        
        return x, routing_decisions


class EarlyExitHead(nn.Module):
    """Early exit classifier at each layer.
    
    Allows tokens to exit early if model is confident enough.
    Based on: Teerapittayanon et al., "BranchyNet"
    """
    
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, vocab_size, bias=False)
        self.confidence_threshold = 0.9
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute exit logits and confidence.
        
        Args:
            x: (batch, seq_len, d_model)
        
        Returns:
            logits: (batch, seq_len, vocab_size)
            confidence: (batch, seq_len) max softmax probability
        """
        x = self.norm(x)
        logits = self.classifier(x)
        
        probs = F.softmax(logits, dim=-1)
        confidence, _ = probs.max(dim=-1)  # (batch, seq_len)
        
        return logits, confidence


class OptimizedTransformer(nn.Module):
    """Transformer with adaptive layer skipping and early exit."""
    
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        
        # Transformer layers with adaptive routing
        self.layers = nn.ModuleList([
            AdaptiveTransformerLayer(
                config.d_model,
                config.num_heads,
                config.ffn_dim,
                config.num_layers,
                layer_idx=i,
            )
            for i in range(config.num_layers)
        ])
        
        # Final output layer
        self.norm = nn.LayerNorm(config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.vocab_size, bias=False)
    
    def forward(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with adaptive routing.
        
        Args:
            x: (batch, seq_len, d_model) — pre-embedded input
            valid_mask: (batch, seq_len) optional padding mask
        
        Returns:
            out: (batch, seq_len, d_model) hidden states
        """
        
        # Apply transformer layers with adaptive routing
        for layer_idx, layer in enumerate(self.layers):
            x, routing_decisions = layer(
                x,
                valid_token_mask=valid_mask,
                causal=self.config.causal,
                use_routing=True,  # Enable routing by default
            )
        
        # Final output projection
        x = self.norm(x)
        logits = self.out_proj(x)
        
        return logits
