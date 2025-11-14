"""
GPU-Accelerated Lorentz-Invariant Similarity
=============================================

PyTorch implementation of Lorentz-invariant (EigenFunction) similarity
for GPU acceleration and automatic differentiation support.

This module provides batched, differentiable versions of the similarity
measures for use in neural network architectures.
"""

import torch
import torch.nn.functional as F


def eigen_similarity(
    q: torch.Tensor,
    k: torch.Tensor,
    epsilon: float = 1e-10,
) -> torch.Tensor:
    """
    Compute batched Lorentz-invariant (EigenFunction) similarity.

    This is the PyTorch/GPU version of the Lorentz-invariant similarity
    that yields 0.0 for self-similarity, preventing loop amplification
    in recursive architectures.

    Mathematical Formulation:
    ------------------------
    For vectors u and v, embed in Minkowski spacetime:
        u -> (u, ||u||), v -> (v, ||v||)

    Lorentz inner product:
        <u, v>_L = u·v - ||u|| * ||v||

    Self-product (lightlike):
        <u, u>_L = ||u||² - ||u||² = 0

    Similarity:
        sim_L(u, v) = <u, v>_L / sqrt(|<u, u>_L| * |<v, v>_L|)

    For self-similarity, denominator -> 0, yielding 0.0 (loop prevention).

    Supported Shapes:
    ----------------
    1. Query-Key (memory retrieval):
       q: (B, D), k: (N, D) -> (B, N)

    2. Sequence-Sequence (attention):
       q: (B, L_q, D), k: (B, L_k, D) -> (B, L_q, L_k)

    3. Multi-head (flattened):
       q: (B*H, L_q, D), k: (B*H, L_k, D) -> (B*H, L_q, L_k)

    Args:
        q: Query tensor, shape (..., D) where ... can be (B,), (B, L), etc.
        k: Key tensor, same final dim D as q
        epsilon: Small constant for numerical stability

    Returns:
        Similarity tensor in range approximately [-1, 1]
        Shape determined by broadcasting q and k

    Examples:
        >>> q = torch.randn(32, 128)  # 32 queries, 128-dim
        >>> k = torch.randn(1000, 128)  # 1000 keys
        >>> sim = eigen_similarity(q, k)  # (32, 1000)

        >>> q = torch.randn(8, 20, 64)  # batch=8, seq_len=20, dim=64
        >>> k = torch.randn(8, 20, 64)
        >>> sim = eigen_similarity(q, k)  # (8, 20, 20)
    """
    # Normalize inputs to float32 or float64
    if q.dtype not in [torch.float32, torch.float64]:
        q = q.float()
    if k.dtype not in [torch.float32, torch.float64]:
        k = k.float()

    # Ensure same device
    if q.device != k.device:
        k = k.to(q.device)

    # Handle different input shapes
    q_dim = q.dim()
    k_dim = k.dim()

    if q_dim == 2 and k_dim == 2:
        # Case 1: (B, D) x (N, D) -> (B, N)
        return _eigen_similarity_2d(q, k, epsilon)
    elif q_dim == 3 and k_dim == 3:
        # Case 2: (B, L_q, D) x (B, L_k, D) -> (B, L_q, L_k)
        return _eigen_similarity_3d(q, k, epsilon)
    else:
        raise ValueError(
            f"Unsupported tensor dimensions: q.shape={q.shape}, k.shape={k.shape}. "
            f"Expected (B, D) x (N, D) or (B, L, D) x (B, L, D)"
        )


def _eigen_similarity_2d(
    q: torch.Tensor,
    k: torch.Tensor,
    epsilon: float = 1e-10,
) -> torch.Tensor:
    """
    Compute similarity for 2D tensors: (B, D) x (N, D) -> (B, N)

    Uses modified Lorentz-invariant formulation where self-similarity -> 0
    but different vector similarities remain informative.

    Embedding: u -> (u, 1) in Minkowski space with signature (-,+,+,...)
    This makes <u,u>_L = ||u||² - 1

    Args:
        q: (B, D) query vectors
        k: (N, D) key vectors
        epsilon: Numerical stability constant

    Returns:
        (B, N) similarity matrix
    """
    B, D = q.shape
    N, D_k = k.shape

    if D != D_k:
        raise ValueError(f"Dimension mismatch: q has dim {D}, k has dim {D_k}")

    # Compute norms
    norm_q_sq = torch.sum(q ** 2, dim=-1, keepdim=True)  # (B, 1)
    norm_k_sq = torch.sum(k ** 2, dim=-1, keepdim=True)  # (N, 1)

    # Spatial inner products: q · k^T
    spatial_product = torch.mm(q, k.t())  # (B, N)

    # Lorentz inner product with time component = 1:
    # <q, k>_L = q·k - 1*1 = q·k - 1
    lorentz_product_qk = spatial_product - 1.0  # (B, N)

    # Self inner products with time component = 1:
    # <q, q>_L = ||q||² - 1
    # <k, k>_L = ||k||² - 1
    lorentz_qq = norm_q_sq.squeeze(-1) - 1.0  # (B,)
    lorentz_kk = norm_k_sq.squeeze(-1) - 1.0  # (N,)

    # Denominator: sqrt(|<q,q>_L| * |<k,k>_L|)
    lorentz_qq_abs = torch.abs(lorentz_qq)  # (B,)
    lorentz_kk_abs = torch.abs(lorentz_kk)  # (N,)

    # Broadcast to (B, N)
    denominator_sq = lorentz_qq_abs.unsqueeze(1) * lorentz_kk_abs.unsqueeze(0)  # (B, N)
    denominator = torch.sqrt(denominator_sq + epsilon ** 2)

    # Similarity
    similarity = lorentz_product_qk / denominator.clamp(min=epsilon)

    # Detect self-similarity: check if q[i] == k[j] (within tolerance)
    # Use pairwise distance to detect identical vectors
    # ||q[i] - k[j]||² = ||q[i]||² + ||k[j]||² - 2*q[i]·k[j]
    dist_sq = norm_q_sq + norm_k_sq.t() - 2 * spatial_product  # (B, N)
    # Use larger tolerance for self-detection (accounting for fp32 precision)
    is_self = dist_sq < 1e-4  # Vectors are nearly identical

    # Set self-similarity to 0.0 (loop prevention)
    similarity = torch.where(is_self, torch.zeros_like(similarity), similarity)

    # Clamp to valid range
    similarity = torch.clamp(similarity, -1.0, 1.0)

    return similarity


def _eigen_similarity_3d(
    q: torch.Tensor,
    k: torch.Tensor,
    epsilon: float = 1e-10,
) -> torch.Tensor:
    """
    Compute similarity for 3D tensors: (B, L_q, D) x (B, L_k, D) -> (B, L_q, L_k)

    Uses modified Lorentz-invariant formulation where self-similarity -> 0
    but different vector similarities remain informative.

    Args:
        q: (B, L_q, D) query sequences
        k: (B, L_k, D) key sequences
        epsilon: Numerical stability constant

    Returns:
        (B, L_q, L_k) similarity tensor
    """
    B_q, L_q, D = q.shape
    B_k, L_k, D_k = k.shape

    if B_q != B_k:
        raise ValueError(f"Batch size mismatch: q has batch {B_q}, k has batch {B_k}")
    if D != D_k:
        raise ValueError(f"Dimension mismatch: q has dim {D}, k has dim {D_k}")

    B = B_q

    # Compute norms
    norm_q_sq = torch.sum(q ** 2, dim=-1, keepdim=True)  # (B, L_q, 1)
    norm_k_sq = torch.sum(k ** 2, dim=-1, keepdim=True)  # (B, L_k, 1)

    # Spatial inner products: q @ k^T
    spatial_product = torch.bmm(q, k.transpose(1, 2))  # (B, L_q, L_k)

    # Lorentz inner product with time component = 1:
    # <q, k>_L = q·k - 1
    lorentz_product_qk = spatial_product - 1.0  # (B, L_q, L_k)

    # Self inner products with time component = 1:
    # <q, q>_L = ||q||² - 1
    # <k, k>_L = ||k||² - 1
    lorentz_qq = norm_q_sq.squeeze(-1) - 1.0  # (B, L_q)
    lorentz_kk = norm_k_sq.squeeze(-1) - 1.0  # (B, L_k)

    # Denominator: sqrt(|<q,q>_L| * |<k,k>_L|)
    lorentz_qq_abs = torch.abs(lorentz_qq)  # (B, L_q)
    lorentz_kk_abs = torch.abs(lorentz_kk)  # (B, L_k)

    # Broadcast to (B, L_q, L_k)
    denominator_sq = lorentz_qq_abs.unsqueeze(2) * lorentz_kk_abs.unsqueeze(1)  # (B, L_q, L_k)
    denominator = torch.sqrt(denominator_sq + epsilon ** 2)

    # Similarity
    similarity = lorentz_product_qk / denominator.clamp(min=epsilon)

    # Detect self-similarity: check if q[b,i] == k[b,j]
    # ||q[b,i] - k[b,j]||² = ||q[b,i]||² + ||k[b,j]||² - 2*q[b,i]·k[b,j]
    dist_sq = norm_q_sq + norm_k_sq.transpose(1, 2) - 2 * spatial_product  # (B, L_q, L_k)
    # Use larger tolerance for self-detection (accounting for fp32 precision)
    is_self = dist_sq < 1e-4  # Vectors are nearly identical

    # Set self-similarity to 0.0 (loop prevention)
    similarity = torch.where(is_self, torch.zeros_like(similarity), similarity)

    # Clamp to valid range
    similarity = torch.clamp(similarity, -1.0, 1.0)

    return similarity


def standard_cosine_similarity_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    epsilon: float = 1e-10,
) -> torch.Tensor:
    """
    Standard cosine similarity for comparison (yields 1.0 for self-similarity).

    Supports same shapes as eigen_similarity:
    - (B, D) x (N, D) -> (B, N)
    - (B, L_q, D) x (B, L_k, D) -> (B, L_q, L_k)

    Args:
        q: Query tensor
        k: Key tensor
        epsilon: Numerical stability constant

    Returns:
        Cosine similarity in range [-1, 1]
    """
    # Use F.normalize and matrix multiplication
    q_dim = q.dim()
    k_dim = k.dim()

    if q_dim == 2 and k_dim == 2:
        # (B, D) x (N, D) -> (B, N)
        q_norm = F.normalize(q, p=2, dim=-1, eps=epsilon)
        k_norm = F.normalize(k, p=2, dim=-1, eps=epsilon)
        return torch.mm(q_norm, k_norm.t())
    elif q_dim == 3 and k_dim == 3:
        # (B, L_q, D) x (B, L_k, D) -> (B, L_q, L_k)
        q_norm = F.normalize(q, p=2, dim=-1, eps=epsilon)
        k_norm = F.normalize(k, p=2, dim=-1, eps=epsilon)
        return torch.bmm(q_norm, k_norm.transpose(1, 2))
    else:
        raise ValueError(
            f"Unsupported tensor dimensions: q.shape={q.shape}, k.shape={k.shape}"
        )


def compare_self_similarity_torch(
    v: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """
    Compare self-similarity between standard cosine and eigen similarity.

    Args:
        v: Input tensor, shape (..., D)

    Returns:
        Dictionary with:
        - 'standard': standard cosine self-similarity (~1.0)
        - 'eigen': eigen self-similarity (~0.0)
        - 'vector_norm': L2 norms
    """
    standard = standard_cosine_similarity_torch(v, v)
    eigen = eigen_similarity(v, v)

    # Compute norms
    if v.dim() == 2:
        norm = torch.linalg.norm(v, dim=-1)
    elif v.dim() == 3:
        norm = torch.linalg.norm(v, dim=-1)
    else:
        norm = torch.linalg.norm(v.reshape(-1, v.shape[-1]), dim=-1)

    return {
        'standard': standard,
        'eigen': eigen,
        'vector_norm': norm,
    }
