import torch
import torch.nn as nn
import torch.nn.functional as F

from gpu_similarity import eigen_similarity


class EigenAttention(nn.Module):
    """
    Multi-head attention where query-key interaction is given by
    Lorentz-invariant EigenFunction similarity instead of dot product.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        bias: bool = True,
        sim_scale: float = 4.0,
        mask_negative: bool = True,
        negative_floor: float = -10.0,
        loop_epsilon: float = 1e-3,
        causal: bool = False,
    ) -> None:
        """
        Args:
            dim: Model dimension.
            num_heads: Number of attention heads.
            bias: Whether to use bias in linear projections.
            sim_scale: Scale factor applied to similarity logits.
            mask_negative: If True, clamp negative logits down to negative_floor.
            negative_floor: Minimum value for negative logits when mask_negative is True.
            loop_epsilon: Treat |sim| < loop_epsilon as self/lightlike and set to 0.
            causal: If True, apply causal mask (no attending to future tokens).
        """
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.W_q = nn.Linear(dim, dim, bias=bias)
        self.W_k = nn.Linear(dim, dim, bias=bias)
        self.W_v = nn.Linear(dim, dim, bias=bias)
        self.W_o = nn.Linear(dim, dim, bias=bias)

        self.sim_scale = sim_scale
        self.mask_negative = mask_negative
        self.negative_floor = negative_floor
        self.loop_epsilon = loop_epsilon
        self.causal = causal

    def _reshape_to_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        (B, L, D) -> (B*H, L, d_head)
        """
        B, L, D = x.shape
        return (
            x.view(B, L, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .reshape(B * self.num_heads, L, self.head_dim)
        )

    def _reshape_from_heads(self, x: torch.Tensor, B: int, L: int) -> torch.Tensor:
        """
        (B*H, L, d_head) -> (B, L, D)
        """
        return (
            x.view(B, self.num_heads, L, self.head_dim)
            .transpose(1, 2)
            .reshape(B, L, self.dim)
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, L, D) input sequence; first component of each vector is timelike.
            attn_mask:
                Optional additive mask broadcastable to (B, 1, L, L) or (B, L, L).
                Typically contains 0 for allowed and -inf (or large negative) for masked.

        Returns:
            out: (B, L, D) attended outputs.
            attn: (B, H, L, L) attention weights.
        """
        if x.ndim != 3:
            raise ValueError(f"EigenAttention expects x of shape (B, L, D), got {x.shape}")
        B, L, D = x.shape
        if D != self.dim:
            raise ValueError(f"Expected dim={self.dim}, got {D}")

        device = x.device

        # Project to Q, K, V
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        # Multi-head reshape
        q = self._reshape_to_heads(q)  # (B*H, L, d_head)
        k = self._reshape_to_heads(k)  # (B*H, L, d_head)
        v = self._reshape_to_heads(v)  # (B*H, L, d_head)

        # Eigen similarity per head: (B*H, L_q, L_k)
        sim = eigen_similarity(q, k)   # expected in [-1, 1]

        # loop prevention: attenuate near-self/lightlike connections
        sim = torch.where(
            sim.abs() < self.loop_epsilon,
            torch.zeros_like(sim),
            sim,
        )

        # scale logits
        logits = self.sim_scale * sim  # (B*H, L, L)

        # optional suppression of negative (disconnected) interactions
        if self.mask_negative:
            logits = torch.where(
                logits < 0.0,
                logits.clamp(min=self.negative_floor),
                logits,
            )

        # causal masking: prevent attending to future positions
        if self.causal:
            causal_mask = torch.triu(
                torch.ones(L, L, dtype=torch.bool, device=device),
                diagonal=1,
            )
            logits = logits.masked_fill(causal_mask.unsqueeze(0), float("-inf"))

        # external attention mask
        if attn_mask is not None:
            # Support shapes:
            #   (B, L, L)  -> expand to (B*H, L, L)
            #   (B, 1, L, L) -> expand to (B*H, 1, L, L) then squeeze
            if attn_mask.dim() == 3:
                # (B, L, L)
                attn_mask_expanded = attn_mask.repeat_interleave(self.num_heads, dim=0)
                logits = logits + attn_mask_expanded
            elif attn_mask.dim() == 4:
                # (B, 1, L, L) or (B, H, L, L)
                if attn_mask.shape[1] == 1:
                    attn_mask_expanded = attn_mask.repeat(1, self.num_heads, 1, 1)
                else:
                    attn_mask_expanded = attn_mask
                attn_mask_expanded = attn_mask_expanded.reshape(
                    B * self.num_heads, L, L
                )
                logits = logits + attn_mask_expanded
            else:
                raise ValueError(
                    f"Unsupported attn_mask shape: {attn_mask.shape}, expected 3D or 4D mask."
                )

        # attention weights
        attn = F.softmax(logits, dim=-1)  # (B*H, L, L)

        # value aggregation
        out = torch.bmm(attn, v)  # (B*H, L, d_head)

        # reshape back to (B, L, D)
        out = self._reshape_from_heads(out, B, L)
        out = self.W_o(out)

        # reshape attention to (B, H, L, L)
        attn = attn.view(B, self.num_heads, L, L)

        return out, attn
