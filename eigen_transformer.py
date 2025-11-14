import torch
import torch.nn as nn

from eigen_attention import EigenAttention


class EigenTransformerBlock(nn.Module):
    """
    A minimal transformer block that uses EigenAttention instead of
    dot-product attention, followed by a standard MLP.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_kwargs: dict | None = None,
    ) -> None:
        """
        Args:
            dim: Model dimension.
            num_heads: Number of attention heads.
            mlp_ratio: Expansion factor for MLP hidden size.
            dropout: Dropout rate for residual paths and MLP.
            attn_kwargs: Extra keyword arguments for EigenAttention
                         (e.g., {"causal": True}).
        """
        super().__init__()
        if attn_kwargs is None:
            attn_kwargs = {}

        self.dim = dim
        self.num_heads = num_heads

        self.norm1 = nn.LayerNorm(dim)
        self.attn = EigenAttention(dim=dim, num_heads=num_heads, **attn_kwargs)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, L, D) input sequence.
            attn_mask: Optional additive mask compatible with EigenAttention.

        Returns:
            x: (B, L, D) output.
        """
        if x.ndim != 3:
            raise ValueError(f"EigenTransformerBlock expects (B, L, D), got {x.shape}")

        # Attention + residual
        h = self.norm1(x)
        attn_out, _ = self.attn(h, attn_mask=attn_mask)
        x = x + self.dropout1(attn_out)

        # MLP + residual
        h = self.norm2(x)
        x = x + self.dropout2(self.mlp(h))

        return x
