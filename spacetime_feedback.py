from __future__ import annotations

import torch
import torch.nn as nn

from eigen_attention import EigenAttention
from standard_attention import StandardAttention


class SpacetimeFeedbackBlock(nn.Module):
    """
    Spacetime-structured feedback architecture using Minkowski causal structure.

    Architecture:
    - Timelike branch: Causal/sequential computation (inside light cone, ds² < 0)
    - Spacelike branch: Acausal/parallel computation (outside light cone, ds² > 0)
    - Lightlike monitor: Equilibrium detector (on light cone, ds² = 0)

    The lightlike layer detects when timelike and spacelike processing are
    out of balance, preventing causal loops (too timelike) or disconnection
    (too spacelike).

    Physical interpretation:
    - Timelike dominance → Over-sequential → Causal loops
    - Spacelike dominance → Over-parallel → Disconnected computation
    - Lightlike equilibrium → Balanced → Stable computation
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        loop_epsilon: float = 1e-3,
        feedback_strength: float = 0.5,
    ) -> None:
        """
        Args:
            dim: Model dimension.
            num_heads: Number of attention heads (split between timelike/spacelike).
            mlp_ratio: Expansion factor for MLP.
            dropout: Dropout rate.
            loop_epsilon: Loop prevention threshold for lightlike monitor.
            feedback_strength: Correction signal strength.
        """
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.feedback_strength = feedback_strength
        self.loop_epsilon = loop_epsilon

        heads_per_branch = max(1, num_heads // 2)

        # ===== Timelike Branch (Causal, Sequential) =====
        # Uses standard Euclidean attention with causal masking
        # Represents temporal processing within the light cone
        self.norm_timelike = nn.LayerNorm(dim)
        self.timelike_branch = StandardAttention(
            dim=dim,
            num_heads=heads_per_branch,
            dropout=dropout,
            causal=True,  # Causal = timelike
        )

        # ===== Spacelike Branch (Acausal, Parallel) =====
        # Uses standard Euclidean attention without causal masking
        # Represents spatial processing outside the light cone
        self.norm_spacelike = nn.LayerNorm(dim)
        self.spacelike_branch = StandardAttention(
            dim=dim,
            num_heads=heads_per_branch,
            dropout=dropout,
            causal=False,  # Non-causal = spacelike
        )

        # ===== Lightlike Monitor (Null Boundary, ds² = 0) =====
        # Uses Lorentz-invariant attention where self-similarity ≈ 0
        # Sits on the lightlike boundary to detect imbalance
        self.norm_lightlike = nn.LayerNorm(dim * 2)
        self.lightlike_monitor = EigenAttention(
            dim=dim * 2,
            num_heads=num_heads,
            loop_epsilon=loop_epsilon,  # Suppresses near-null similarities
            causal=False,  # Monitors full context
        )

        # ===== Spacetime Interval Detector =====
        # Computes effective ds² = timelike² - spacelike²
        # Positive → Too spacelike, Negative → Too timelike, Zero → Lightlike (balanced)
        self.interval_detector = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
            nn.Tanh(),  # Output in [-1, 1]
        )

        # ===== Feedback Correction Network =====
        self.feedback_head = nn.Linear(dim * 2, dim)

        # ===== Standard MLP =====
        self.norm_mlp = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def compute_spacetime_interval(
        self,
        timelike_out: torch.Tensor,
        spacelike_out: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute effective spacetime interval ds² = timelike² - spacelike².

        Args:
            timelike_out: (B, L, D) output from timelike branch
            spacelike_out: (B, L, D) output from spacelike branch

        Returns:
            interval: (B,) spacetime interval
                > 0: Spacelike dominance (too parallel/disconnected)
                < 0: Timelike dominance (too sequential/looping)
                ≈ 0: Lightlike (balanced/equilibrium)
            imbalance: (B,) absolute imbalance magnitude in [0, 1]
        """
        # Concatenate for joint analysis
        combined = torch.cat([timelike_out, spacelike_out], dim=-1)  # (B, L, D*2)

        # Compute interval: ds² ∝ spacelike² - timelike²
        # (using Minkowski signature: -,+,+,+)
        interval = self.interval_detector(combined)  # (B, L, 1)
        interval_score = interval.mean(dim=1).squeeze(-1)  # (B,)

        # Imbalance is magnitude of deviation from lightlike (ds²=0)
        imbalance = interval_score.abs()  # (B,)

        return interval_score, imbalance

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        """
        Args:
            x: (B, L, D) input sequence.
            attn_mask: Optional attention mask.
            return_diagnostics: If True, return spacetime interval and imbalance.

        Returns:
            output: (B, L, D) output sequence.
            diagnostics: dict with 'interval', 'imbalance', 'causal_type' (if requested)
        """
        if x.ndim != 3:
            raise ValueError(f"SpacetimeFeedbackBlock expects (B, L, D), got {x.shape}")

        B, L, D = x.shape

        # ===== Timelike Processing (Causal) =====
        h_time = self.norm_timelike(x)
        timelike_out, timelike_attn = self.timelike_branch(h_time, attn_mask=attn_mask)

        # ===== Spacelike Processing (Acausal) =====
        h_space = self.norm_spacelike(x)
        spacelike_out, spacelike_attn = self.spacelike_branch(h_space, attn_mask=attn_mask)

        # ===== Compute Spacetime Interval =====
        interval, imbalance = self.compute_spacetime_interval(timelike_out, spacelike_out)

        # ===== Lightlike Monitor (Equilibrium Detection) =====
        combined = torch.cat([timelike_out, spacelike_out], dim=-1)  # (B, L, D*2)
        combined_norm = self.norm_lightlike(combined)

        # Lorentz monitor sits on lightlike boundary (ds²=0)
        monitored, monitor_attn = self.lightlike_monitor(combined_norm)

        # ===== Feedback Correction =====
        # Generate correction from lightlike monitor
        correction = self.feedback_head(monitored)  # (B, L, D)

        # Scale correction by imbalance magnitude
        # High imbalance → strong correction to restore equilibrium
        correction_weight = imbalance.unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
        correction_scaled = correction * correction_weight * self.feedback_strength

        # ===== Combine Branches =====
        # At equilibrium (lightlike): timelike ≈ spacelike, minimal correction
        # Imbalanced: correction restores balance
        attn_out = timelike_out + spacelike_out + correction_scaled

        # Residual connection
        x = x + self.dropout(attn_out)

        # ===== MLP =====
        h = self.norm_mlp(x)
        x = x + self.dropout(self.mlp(h))

        if return_diagnostics:
            # Classify causal type based on interval
            causal_type = torch.where(
                interval.abs() < self.loop_epsilon,
                torch.zeros_like(interval),  # 0 = lightlike (balanced)
                torch.where(
                    interval > 0,
                    torch.ones_like(interval),  # 1 = spacelike dominant
                    -torch.ones_like(interval),  # -1 = timelike dominant
                ),
            )

            diagnostics = {
                "interval": interval,  # ds² value
                "imbalance": imbalance,  # |ds²|
                "causal_type": causal_type,  # -1: timelike, 0: lightlike, 1: spacelike
                "timelike_attn": timelike_attn,
                "spacelike_attn": spacelike_attn,
                "monitor_attn": monitor_attn,
            }
            return x, diagnostics

        return x


def interpret_causal_type(causal_type: torch.Tensor) -> str:
    """
    Interpret the causal type value.

    Args:
        causal_type: Scalar tensor with value -1, 0, or 1

    Returns:
        Human-readable interpretation
    """
    val = causal_type.item()
    if abs(val) < 1e-6:
        return "Lightlike (Balanced/Equilibrium) - ds² ≈ 0"
    elif val > 0:
        return "Spacelike Dominant (Too Parallel/Disconnected) - ds² > 0"
    else:
        return "Timelike Dominant (Too Sequential/Looping) - ds² < 0"
