"""Composite transformer for optimization phases 1-3.

Phase 1: scaled dot-product attention and external torch.compile support.
Phase 2: grouped-query attention and a fused gated FFN.
Phase 3: optional adaptive layer routing for a trained router.

The model intentionally keeps the benchmark's pre-embedded tensor interface:
    OptimizedTransformer(config)(x, valid_token_mask) -> [B, S, d_model]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FusedGLU(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, hidden_dim * 2)
        self.out = nn.Linear(hidden_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.proj(x).chunk(2, dim=-1)
        return self.out(value * F.silu(gate))


class GQASDPA(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int,
        max_seq_len: int,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.head_dim)
        self.out_proj = nn.Linear(d_model, d_model)
        self.register_buffer(
            "inv_freq",
            1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim)),
            persistent=False,
        )
        self.max_seq_len = max_seq_len

    def _heads(self, x: torch.Tensor, heads: int) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        return x.view(batch, seq_len, heads, self.head_dim).transpose(1, 2)

    def _rope(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[-2]
        if seq_len > self.max_seq_len:
            raise ValueError("sequence length exceeds max_seq_len")
        positions = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        angles = torch.outer(positions, self.inv_freq).to(dtype=x.dtype)
        cos = angles.cos()[None, None, :, :]
        sin = angles.sin()[None, None, :, :]
        first, second = x[..., ::2], x[..., 1::2]
        rotated = torch.stack((first * cos - second * sin, first * sin + second * cos), dim=-1)
        return rotated.flatten(-2)

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: torch.Tensor | None,
        causal: bool,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        q = self._rope(self._heads(self.q_proj(x), self.num_heads))
        k = self._rope(self._heads(self.k_proj(x), self.num_kv_heads))
        v = self._heads(self.v_proj(x), self.num_kv_heads)
        repeat_count = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(repeat_count, dim=1)
        v = v.repeat_interleave(repeat_count, dim=1)

        attn_mask = None
        if valid_token_mask is not None:
            invalid = ~valid_token_mask[:, None, None, :]
            attn_mask = torch.zeros(
                batch, 1, seq_len, seq_len, device=x.device, dtype=x.dtype
            ).masked_fill(invalid, float("-inf"))
        output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=causal, scale=self.scale
        )
        output = output.transpose(1, 2).reshape(batch, seq_len, -1)
        output = self.out_proj(output)
        if valid_token_mask is not None:
            output = output.masked_fill(~valid_token_mask[..., None], 0)
        return output


class AdaptiveRouter(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(d_model, max(1, d_model // 2)),
            nn.SiLU(),
            nn.Linear(max(1, d_model // 2), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.network(x.mean(dim=1))).squeeze(-1)


class FinalTransformerLayer(nn.Module):
    def __init__(self, config, num_kv_heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model)
        self.attention = GQASDPA(
            config.d_model,
            config.num_heads,
            num_kv_heads,
            getattr(config, "max_seq_len", max(2048, config.seq_len)),
        )
        self.norm2 = nn.LayerNorm(config.d_model)
        self.ffn = FusedGLU(config.d_model, config.ffn_dim)
        self.router = AdaptiveRouter(config.d_model)

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor | None, config) -> torch.Tensor:
        residual = x
        attention = self.attention(self.norm1(x), valid_mask, config.causal)
        if config.routing_enabled and not self.training:
            keep = (1.0 - self.router(x)).clamp(0.0, 1.0)[:, None, None]
            x = residual + keep * attention
        else:
            x = residual + attention
        x = x + self.ffn(self.norm2(x))
        if valid_mask is not None:
            x = x.masked_fill(~valid_mask[..., None], 0)
        return x


class OptimizedTransformer(nn.Module):
    """Final composite model; torch.compile is applied by the benchmark."""

    def __init__(self, config) -> None:
        super().__init__()
        num_kv_heads = getattr(config, "num_kv_heads", max(1, config.num_heads // 4))
        if config.num_heads % num_kv_heads != 0:
            raise ValueError("num_kv_heads must divide num_heads")
        config.routing_enabled = getattr(config, "routing_enabled", False)
        self.config = config
        self.layers = nn.ModuleList(
            [FinalTransformerLayer(config, num_kv_heads) for _ in range(config.num_layers)]
        )
        self.final_norm = nn.LayerNorm(config.d_model)

    def forward(
        self, x: torch.Tensor, valid_token_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, valid_token_mask, self.config)
        x = self.final_norm(x)
        if valid_token_mask is not None:
            x = x.masked_fill(~valid_token_mask[..., None], 0)
        return x
