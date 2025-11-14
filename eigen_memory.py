import torch
import torch.nn as nn
import torch.nn.functional as F

from gpu_similarity import eigen_similarity


class EigenMemory(nn.Module):
    """
    EigenFunction-based memory module using Lorentz-invariant similarity
    for retrieval, with loop prevention.

    Assumptions:
        - Hidden vectors have shape (B, D).
        - First component is timelike; others are spacelike.
        - eigen_similarity(q, k) returns (B, N) similarities in [-1, 1].
    """

    def __init__(
        self,
        dim: int,
        max_mem_slots: int = 4096,
        k_top: int = 32,
        sim_threshold: float = 0.0,
        loop_epsilon: float = 1e-3,
        decay: float = 0.99,
        device: str | torch.device | None = None,
    ) -> None:
        """
        Args:
            dim: Embedding dimension.
            max_mem_slots: Maximum number of memory entries.
            k_top: Top-k neighbors to use in retrieval.
            sim_threshold: Ignore similarities <= this value (for negative/disconnected).
            loop_epsilon: Treat |sim| < loop_epsilon as self/lightlike and ignore.
            decay: Temporal decay factor for older entries (0 < decay <= 1).
            device: Device to place buffers on; if None, use current default device.
        """
        super().__init__()

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)

        self.dim = dim
        self.max_mem_slots = max_mem_slots
        self.k_top = k_top
        self.sim_threshold = sim_threshold
        self.loop_epsilon = loop_epsilon
        self.decay = decay

        self.register_buffer(
            "mem",
            torch.zeros(max_mem_slots, dim, device=device),
            persistent=True,
        )
        self.register_buffer(
            "age",
            torch.zeros(max_mem_slots, device=device),
            persistent=True,
        )
        self.register_buffer(
            "count",
            torch.zeros(1, dtype=torch.long, device=device),
            persistent=True,
        )

    @property
    def device(self) -> torch.device:
        return self.mem.device

    @torch.no_grad()
    def write(self, x: torch.Tensor) -> None:
        """
        Write new states into memory.

        Args:
            x: (B, D) tensor of new states.
        """
        if x.ndim != 2:
            raise ValueError(f"write expects (B, D), got {x.shape}")
        B, D = x.shape
        if D != self.dim:
            raise ValueError(f"Expected dim={self.dim}, got {D}")

        x = x.to(self.device)

        idx = int(self.count.item())
        end = min(idx + B, self.max_mem_slots)
        n = end - idx

        if n <= 0:
            # Simple ring-buffer overwrite policy when full
            idx = 0
            end = min(B, self.max_mem_slots)
            n = end - idx

        if n > 0:
            self.mem[idx:end] = x[:n]
            self.age[idx:end] = 0.0
            self.count[0] = min(self.max_mem_slots, idx + n)

        # increase age for all slots
        self.age += 1.0

    def forward(
        self,
        q: torch.Tensor,
        return_weights: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]] | torch.Tensor:
        """
        Retrieve from memory given queries.

        Args:
            q: (B, D) query states.
            return_weights: If True, also return (attn, idx_topk).

        Returns:
            retrieved: (B, D) retrieved summaries.
            optionally (attn, idx_topk):
                attn: (B, k)
                idx_topk: (B, k)
        """
        if q.ndim != 2:
            raise ValueError(f"forward expects q of shape (B, D), got {q.shape}")
        B, D = q.shape
        if D != self.dim:
            raise ValueError(f"Expected dim={self.dim}, got {D}")

        q = q.to(self.device)
        N = int(self.count.item())
        if N == 0:
            retrieved = torch.zeros_like(q)
            return (retrieved, (None, None)) if return_weights else retrieved

        mem = self.mem[:N]  # (N, D)
        age = self.age[:N]  # (N,)

        # eigen_similarity should support (B, D) x (N, D) -> (B, N)
        sim = eigen_similarity(q, mem)  # (B, N), in [-1, 1]

        # loop prevention: remove near-self/lightlike connections
        sim = torch.where(
            sim.abs() < self.loop_epsilon,
            torch.zeros_like(sim),
            sim,
        )

        # ignore strongly negative (disconnected) entries
        sim = sim.masked_fill(sim <= self.sim_threshold, float("-inf"))

        # temporal decay: favor recent entries (age=0 newest)
        # effective weight ∝ decay^age  ∈ (0, 1]
        decay_factor = (self.decay**age).clamp(min=1e-6)  # (N,)
        sim = sim + decay_factor.log().unsqueeze(0)  # add log-decay

        # top-k selection
        k = min(self.k_top, N)
        sim_topk, idx_topk = torch.topk(sim, k, dim=-1)  # (B, k)

        # attention weights over top-k
        attn = F.softmax(sim_topk, dim=-1)  # (B, k)

        # gather memory vectors
        mem_topk = mem[idx_topk]  # (B, k, D)

        retrieved = torch.sum(attn.unsqueeze(-1) * mem_topk, dim=1)  # (B, D)

        if return_weights:
            return retrieved, (attn, idx_topk)
        return retrieved
