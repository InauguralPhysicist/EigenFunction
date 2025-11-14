from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class StandardAttention(nn.Module):
    """
    Standard multi-head attention using Euclidean dot-product similarity.
    This is the baseline that can exhibit oscillation/loops.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        bias: bool = True,
        dropout: float = 0.0,
        causal: bool = False,
    ) -> None:
        """
        Args:
            dim: Model dimension.
            num_heads: Number of attention heads.
            bias: Whether to use bias in linear projections.
            dropout: Dropout rate for attention weights.
            causal: If True, apply causal mask (no attending to future tokens).
        """
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5  # 1/sqrt(d_k)

        self.W_q = nn.Linear(dim, dim, bias=bias)
        self.W_k = nn.Linear(dim, dim, bias=bias)
        self.W_v = nn.Linear(dim, dim, bias=bias)
        self.W_o = nn.Linear(dim, dim, bias=bias)

        self.dropout = nn.Dropout(dropout)
        self.causal = causal

    def _reshape_to_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        (B, L, D) -> (B, H, L, d_head)
        """
        B, L, D = x.shape
        return x.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

    def _reshape_from_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        (B, H, L, d_head) -> (B, L, D)
        """
        B, H, L, d_head = x.shape
        return x.transpose(1, 2).reshape(B, L, self.dim)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, L, D) input sequence.
            attn_mask: Optional additive mask broadcastable to (B, H, L, L).
                       Typically contains 0 for allowed and -inf for masked.

        Returns:
            out: (B, L, D) attended outputs.
            attn: (B, H, L, L) attention weights.
        """
        if x.ndim != 3:
            raise ValueError(f"StandardAttention expects x of shape (B, L, D), got {x.shape}")
        B, L, D = x.shape
        if D != self.dim:
            raise ValueError(f"Expected dim={self.dim}, got {D}")

        device = x.device

        # Project to Q, K, V
        q = self.W_q(x)  # (B, L, D)
        k = self.W_k(x)  # (B, L, D)
        v = self.W_v(x)  # (B, L, D)

        # Multi-head reshape
        q = self._reshape_to_heads(q)  # (B, H, L, d_head)
        k = self._reshape_to_heads(k)  # (B, H, L, d_head)
        v = self._reshape_to_heads(v)  # (B, H, L, d_head)

        # Standard scaled dot-product attention
        # scores: (B, H, L, L)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Causal masking
        if self.causal:
            causal_mask = torch.triu(
                torch.ones(L, L, dtype=torch.bool, device=device),
                diagonal=1,
            )
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        # External attention mask
        if attn_mask is not None:
            scores = scores + attn_mask

        # Attention weights
        attn = F.softmax(scores, dim=-1)  # (B, H, L, L)
        attn = self.dropout(attn)

        # Value aggregation
        out = torch.matmul(attn, v)  # (B, H, L, d_head)

        # Reshape back to (B, L, D)
        out = self._reshape_from_heads(out)
        out = self.W_o(out)

        return out, attn
