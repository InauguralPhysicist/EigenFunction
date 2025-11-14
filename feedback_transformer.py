from __future__ import annotations

import torch
import torch.nn as nn

from eigen_attention import EigenAttention
from standard_attention import StandardAttention


class FeedbackTransformerBlock(nn.Module):
    """
    Hybrid transformer block implementing XOR feedback architecture:

    - XOR_left: Euclidean attention (can oscillate, outputs ~0)
    - XOR_right: Euclidean attention (can oscillate, outputs ~1)
    - XOR_top: Lorentz attention (monitors opposition, detects imbalance)

    The Lorentz top layer uses Minkowski geometry to detect when the
    Euclidean branches are in opposing states (oscillating) and provides
    corrective feedback to stabilize the system.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        loop_epsilon: float = 1e-3,
        feedback_strength: float = 0.5,
        causal: bool = False,
    ) -> None:
        """
        Args:
            dim: Model dimension.
            num_heads: Number of attention heads (split between left/right).
            mlp_ratio: Expansion factor for MLP hidden size.
            dropout: Dropout rate for residual paths.
            loop_epsilon: Loop prevention threshold for Lorentz monitor.
            feedback_strength: Strength of correction signal (0-1).
            causal: If True, apply causal masking.
        """
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.feedback_strength = feedback_strength

        # Split heads between left and right branches
        heads_per_branch = max(1, num_heads // 2)

        # Euclidean branches (can oscillate)
        self.norm_left = nn.LayerNorm(dim)
        self.euclidean_left = StandardAttention(
            dim=dim,
            num_heads=heads_per_branch,
            dropout=dropout,
            causal=causal,
        )

        self.norm_right = nn.LayerNorm(dim)
        self.euclidean_right = StandardAttention(
            dim=dim,
            num_heads=heads_per_branch,
            dropout=dropout,
            causal=causal,
        )

        # Lorentz monitor (detects oscillation/opposition)
        self.norm_monitor = nn.LayerNorm(dim * 2)
        self.lorentz_monitor = EigenAttention(
            dim=dim * 2,  # Monitors concatenation of both branches
            num_heads=num_heads,
            loop_epsilon=loop_epsilon,
            causal=False,  # Monitor sees full context
        )

        # Imbalance detector: measures opposition between branches
        self.imbalance_head = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
            nn.Sigmoid(),  # Output in [0, 1]
        )

        # Feedback correction network
        self.feedback_head = nn.Linear(dim * 2, dim)

        # MLP (standard feedforward)
        self.norm_mlp = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def detect_imbalance(
        self,
        left_out: torch.Tensor,
        right_out: torch.Tensor,
        left_attn: torch.Tensor,
        right_attn: torch.Tensor,
    ) -> torch.Tensor:
        """
        Detect imbalance/oscillation between Euclidean branches.

        Args:
            left_out: (B, L, D) output from left branch
            right_out: (B, L, D) output from right branch
            left_attn: (B, H, L, L) attention weights from left
            right_attn: (B, H, L, L) attention weights from right

        Returns:
            imbalance_score: (B,) scalar imbalance score in [0, 1]
        """
        # Concatenate outputs for monitoring
        combined = torch.cat([left_out, right_out], dim=-1)  # (B, L, D*2)

        # Use imbalance detector
        imbalance = self.imbalance_head(combined)  # (B, L, 1)
        imbalance_score = imbalance.mean(dim=1).squeeze(-1)  # (B,)

        return imbalance_score

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        return_imbalance: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, L, D) input sequence.
            attn_mask: Optional attention mask.
            return_imbalance: If True, also return imbalance scores.

        Returns:
            output: (B, L, D) output sequence.
            imbalance_score: (B,) if return_imbalance=True
        """
        if x.ndim != 3:
            raise ValueError(f"FeedbackTransformerBlock expects (B, L, D), got {x.shape}")

        B, L, D = x.shape

        # ===== Euclidean Computation (XOR_left and XOR_right) =====

        # Left branch (can output ~0)
        h_left = self.norm_left(x)
        left_out, left_attn = self.euclidean_left(h_left, attn_mask=attn_mask)

        # Right branch (can output ~1, opposing left)
        h_right = self.norm_right(x)
        right_out, right_attn = self.euclidean_right(h_right, attn_mask=attn_mask)

        # ===== Lorentz Monitor (XOR_top) =====

        # Concatenate both branches for monitoring
        combined = torch.cat([left_out, right_out], dim=-1)  # (B, L, D*2)
        combined_norm = self.norm_monitor(combined)

        # Lorentz attention monitors for oscillation
        monitored, monitor_attn = self.lorentz_monitor(combined_norm)

        # Detect imbalance between branches
        imbalance_score = self.detect_imbalance(left_out, right_out, left_attn, right_attn)

        # ===== Feedback Correction =====

        # Generate correction signal from Lorentz monitor
        correction = self.feedback_head(monitored)  # (B, L, D)

        # Apply correction proportional to imbalance
        # High imbalance → more correction
        correction_weight = imbalance_score.unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
        correction_scaled = correction * correction_weight * self.feedback_strength

        # Combine branches with feedback
        # When balanced: output ≈ left + right
        # When imbalanced: output includes Lorentz correction
        attn_out = left_out + right_out + correction_scaled

        # Residual connection
        x = x + self.dropout(attn_out)

        # ===== MLP (Standard) =====

        h = self.norm_mlp(x)
        x = x + self.dropout(self.mlp(h))

        if return_imbalance:
            return x, imbalance_score
        return x
