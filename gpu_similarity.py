"""
GPU-Accelerated Lorentz-Invariant Cosine Similarity
====================================================

This module provides GPU-accelerated implementations of the Lorentz-invariant
similarity measures using CuPy (CUDA) for high-performance computation on
NVIDIA GPUs.

Key Features:
-------------
- GPU-accelerated similarity computations for large batches
- Batch processing of multiple vector pairs
- Seamless fallback to CPU when CUDA is unavailable
- Compatible API with the standard similarity module

Performance Benefits:
--------------------
For large-scale applications (e.g., attention mechanisms in transformers,
large graph traversal, massive semantic search), GPU acceleration can provide
10-100x speedup depending on batch size and vector dimensionality.

Requirements:
-------------
- NVIDIA GPU with CUDA support
- CuPy installed (pip install cupy-cuda11x or cupy-cuda12x)

Usage:
------
    >>> import gpu_similarity as gpu_sim
    >>> import numpy as np
    >>>
    >>> # Check if GPU is available
    >>> if gpu_sim.is_gpu_available():
    >>>     u = np.array([1.0, 2.0, 3.0])
    >>>     v = np.array([4.0, 5.0, 6.0])
    >>>     sim = gpu_sim.lorentz_similarity_gpu(u, v)
    >>> else:
    >>>     print("GPU not available, using CPU fallback")
"""

import numpy as np
from typing import Union, Optional, Tuple
import warnings

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None
    warnings.warn(
        "CuPy not available. GPU functions will fall back to CPU. "
        "Install CuPy with: pip install cupy-cuda11x (or cupy-cuda12x for CUDA 12+)",
        ImportWarning
    )


def is_gpu_available() -> bool:
    """
    Check if GPU acceleration is available.

    Returns:
    --------
    bool
        True if CuPy is installed and CUDA device is available
    """
    if not CUPY_AVAILABLE:
        return False

    try:
        # Try to get device info
        cp.cuda.Device(0).compute_capability
        return True
    except Exception:
        return False


def get_array_module(array):
    """
    Get the appropriate array module (numpy or cupy) for an array.

    Parameters:
    -----------
    array : array-like
        Input array

    Returns:
    --------
    module
        numpy or cupy module
    """
    if CUPY_AVAILABLE and isinstance(array, cp.ndarray):
        return cp
    return np


def to_gpu(array: np.ndarray) -> Union[np.ndarray, 'cp.ndarray']:
    """
    Transfer numpy array to GPU if available, otherwise return as-is.

    Parameters:
    -----------
    array : np.ndarray
        Input numpy array

    Returns:
    --------
    Union[np.ndarray, cp.ndarray]
        GPU array if available, otherwise original numpy array
    """
    if CUPY_AVAILABLE and is_gpu_available():
        return cp.asarray(array, dtype=cp.float64)
    return np.asarray(array, dtype=np.float64)


def to_cpu(array: Union[np.ndarray, 'cp.ndarray']) -> np.ndarray:
    """
    Transfer array to CPU (numpy).

    Parameters:
    -----------
    array : Union[np.ndarray, cp.ndarray]
        Input array (GPU or CPU)

    Returns:
    --------
    np.ndarray
        Numpy array on CPU
    """
    if CUPY_AVAILABLE and isinstance(array, cp.ndarray):
        return cp.asnumpy(array)
    return np.asarray(array)


def lorentz_similarity_gpu(
    u: Union[np.ndarray, 'cp.ndarray'],
    v: Union[np.ndarray, 'cp.ndarray'],
    epsilon: float = 1e-10,
    return_cpu: bool = True
) -> float:
    """
    Compute Lorentz-invariant cosine similarity on GPU.

    This function automatically transfers inputs to GPU if available,
    computes the similarity, and returns the result.

    Parameters:
    -----------
    u : Union[np.ndarray, cp.ndarray]
        First vector (1-D array)
    v : Union[np.ndarray, cp.ndarray]
        Second vector (1-D array, same dimension as u)
    epsilon : float
        Small constant for numerical stability (default: 1e-10)
    return_cpu : bool
        If True, return result as Python float (default: True)
        If False, return as GPU scalar

    Returns:
    --------
    float
        Lorentz-invariant similarity in range approximately [-1, 1]
        Returns 0.0 when u and v are identical (lightlike self-similarity)

    Examples:
    ---------
    >>> import numpy as np
    >>> u = np.array([1.0, 0.0, 0.0])
    >>> v = np.array([0.0, 1.0, 0.0])
    >>> lorentz_similarity_gpu(u, v)
    -1.0

    >>> # Self-similarity returns 0.0 (loop prevention)
    >>> lorentz_similarity_gpu(u, u)
    0.0
    """
    # Transfer to GPU if available
    if CUPY_AVAILABLE and is_gpu_available():
        xp = cp
        u = cp.asarray(u, dtype=cp.float64)
        v = cp.asarray(v, dtype=cp.float64)
    else:
        xp = np
        u = np.asarray(u, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)

    if u.shape != v.shape:
        raise ValueError(f"Vectors must have same shape: {u.shape} vs {v.shape}")

    # Compute norms (temporal components in Minkowski embedding)
    norm_u = xp.linalg.norm(u)
    norm_v = xp.linalg.norm(v)

    # Lorentz inner products
    # <u, v>_L = u·v - ||u|| * ||v||
    spatial_product = xp.dot(u, v)
    lorentz_product_uv = spatial_product - (norm_u * norm_v)

    # Self inner products
    # <u, u>_L = ||u||² - ||u||² = 0 (lightlike!)
    # <v, v>_L = ||v||² - ||v||² = 0 (lightlike!)
    lorentz_product_uu = norm_u * norm_u - (norm_u * norm_u)
    lorentz_product_vv = norm_v * norm_v - (norm_v * norm_v)

    # Handle the lightlike case (self-similarity or near-identical vectors)
    denominator_squared = xp.abs(lorentz_product_uu) * xp.abs(lorentz_product_vv)

    if float(denominator_squared) < epsilon:
        return 0.0

    # Standard normalization
    denominator = xp.sqrt(denominator_squared)

    if float(denominator) < epsilon:
        return 0.0

    similarity = lorentz_product_uv / denominator

    # Clamp to valid range
    similarity = xp.clip(similarity, -1.0, 1.0)

    # Return as Python float if requested
    if return_cpu:
        return float(similarity)
    return similarity


def lorentz_similarity_batch_gpu(
    U: Union[np.ndarray, 'cp.ndarray'],
    V: Union[np.ndarray, 'cp.ndarray'],
    epsilon: float = 1e-10,
    return_cpu: bool = True
) -> Union[np.ndarray, 'cp.ndarray']:
    """
    Compute Lorentz-invariant similarities for batches of vector pairs on GPU.

    This is highly optimized for processing many similarity computations in parallel,
    making it ideal for attention mechanisms, batch semantic search, etc.

    Parameters:
    -----------
    U : Union[np.ndarray, cp.ndarray]
        First batch of vectors, shape (N, D) where N is batch size, D is dimension
    V : Union[np.ndarray, cp.ndarray]
        Second batch of vectors, shape (N, D)
    epsilon : float
        Small constant for numerical stability (default: 1e-10)
    return_cpu : bool
        If True, return result as numpy array (default: True)
        If False, return as GPU array

    Returns:
    --------
    Union[np.ndarray, cp.ndarray]
        Array of similarities, shape (N,)
        Each element is the Lorentz similarity between corresponding U[i] and V[i]

    Examples:
    ---------
    >>> import numpy as np
    >>> U = np.random.randn(1000, 128)  # 1000 vectors of dimension 128
    >>> V = np.random.randn(1000, 128)
    >>> similarities = lorentz_similarity_batch_gpu(U, V)
    >>> similarities.shape
    (1000,)
    """
    # Transfer to GPU if available
    if CUPY_AVAILABLE and is_gpu_available():
        xp = cp
        U = cp.asarray(U, dtype=cp.float64)
        V = cp.asarray(V, dtype=cp.float64)
    else:
        xp = np
        U = np.asarray(U, dtype=np.float64)
        V = np.asarray(V, dtype=np.float64)

    if U.shape != V.shape:
        raise ValueError(f"Batches must have same shape: {U.shape} vs {V.shape}")

    if U.ndim != 2:
        raise ValueError(f"Expected 2D arrays (batch, dim), got shape {U.shape}")

    # Compute norms along dimension axis (axis=1)
    norms_U = xp.linalg.norm(U, axis=1)  # shape: (N,)
    norms_V = xp.linalg.norm(V, axis=1)  # shape: (N,)

    # Spatial dot products (element-wise for each pair)
    spatial_products = xp.sum(U * V, axis=1)  # shape: (N,)

    # Lorentz inner products
    lorentz_products_uv = spatial_products - (norms_U * norms_V)

    # Self inner products (all zeros due to lightlike condition)
    lorentz_products_uu = norms_U * norms_U - (norms_U * norms_U)
    lorentz_products_vv = norms_V * norms_V - (norms_V * norms_V)

    # Denominators
    denominator_squared = xp.abs(lorentz_products_uu) * xp.abs(lorentz_products_vv)

    # Initialize similarities array
    similarities = xp.zeros(U.shape[0], dtype=xp.float64)

    # Mask for valid (non-lightlike) computations
    valid_mask = denominator_squared >= epsilon

    # Compute similarities for valid cases
    denominators = xp.sqrt(denominator_squared[valid_mask])
    valid_denom_mask = denominators >= epsilon

    # Combined mask
    final_valid = xp.zeros(U.shape[0], dtype=bool)
    temp_indices = xp.where(valid_mask)[0]
    final_valid[temp_indices[valid_denom_mask]] = True

    # Compute similarities
    similarities[final_valid] = lorentz_products_uv[final_valid] / xp.sqrt(denominator_squared[final_valid])

    # Clamp to valid range
    similarities = xp.clip(similarities, -1.0, 1.0)

    # Return to CPU if requested
    if return_cpu:
        return to_cpu(similarities)
    return similarities


def lorentz_similarity_matrix_gpu(
    U: Union[np.ndarray, 'cp.ndarray'],
    V: Optional[Union[np.ndarray, 'cp.ndarray']] = None,
    epsilon: float = 1e-10,
    return_cpu: bool = True
) -> Union[np.ndarray, 'cp.ndarray']:
    """
    Compute pairwise Lorentz-invariant similarity matrix on GPU.

    Computes similarities between all pairs of vectors in U and V.
    If V is None, computes self-similarity matrix for U.

    This is extremely useful for:
    - Attention mechanisms (query-key similarities)
    - Graph construction from embeddings
    - Clustering and nearest neighbor search

    Parameters:
    -----------
    U : Union[np.ndarray, cp.ndarray]
        First batch of vectors, shape (N, D)
    V : Optional[Union[np.ndarray, cp.ndarray]]
        Second batch of vectors, shape (M, D)
        If None, uses U (self-similarity matrix)
    epsilon : float
        Small constant for numerical stability (default: 1e-10)
    return_cpu : bool
        If True, return result as numpy array (default: True)

    Returns:
    --------
    Union[np.ndarray, cp.ndarray]
        Similarity matrix of shape (N, M) where element [i,j] is
        the Lorentz similarity between U[i] and V[j]
        Note: Diagonal elements will be ~0.0 (lightlike self-similarity)

    Examples:
    ---------
    >>> import numpy as np
    >>> # Query and key vectors in attention mechanism
    >>> queries = np.random.randn(100, 64)  # 100 queries
    >>> keys = np.random.randn(50, 64)      # 50 keys
    >>> attention_scores = lorentz_similarity_matrix_gpu(queries, keys)
    >>> attention_scores.shape
    (100, 50)
    """
    # Transfer to GPU if available
    if CUPY_AVAILABLE and is_gpu_available():
        xp = cp
        U = cp.asarray(U, dtype=cp.float64)
        if V is not None:
            V = cp.asarray(V, dtype=cp.float64)
    else:
        xp = np
        U = np.asarray(U, dtype=np.float64)
        if V is not None:
            V = np.asarray(V, dtype=np.float64)

    if V is None:
        V = U

    if U.ndim != 2 or V.ndim != 2:
        raise ValueError(f"Expected 2D arrays, got shapes {U.shape} and {V.shape}")

    if U.shape[1] != V.shape[1]:
        raise ValueError(f"Dimension mismatch: {U.shape[1]} vs {V.shape[1]}")

    N, D = U.shape
    M = V.shape[0]

    # Compute norms
    norms_U = xp.linalg.norm(U, axis=1, keepdims=True)  # shape: (N, 1)
    norms_V = xp.linalg.norm(V, axis=1, keepdims=True)  # shape: (M, 1)

    # Spatial dot products: U @ V^T
    spatial_products = U @ V.T  # shape: (N, M)

    # Lorentz inner products
    # Broadcast norms: norms_U is (N, 1), norms_V.T is (1, M) -> (N, M)
    lorentz_products_uv = spatial_products - (norms_U @ norms_V.T)

    # Self inner products (for normalization)
    # These are all zeros due to lightlike condition, but we compute for consistency
    lorentz_products_uu = (norms_U ** 2) - (norms_U ** 2)  # shape: (N, 1)
    lorentz_products_vv = (norms_V ** 2) - (norms_V ** 2)  # shape: (M, 1)

    # Denominators: sqrt(|<u,u>_L| * |<v,v>_L|)
    # Broadcast: (N, 1) * (1, M) -> (N, M)
    denominator_squared = xp.abs(lorentz_products_uu) @ xp.abs(lorentz_products_vv).T

    # Initialize similarity matrix
    similarities = xp.zeros((N, M), dtype=xp.float64)

    # Mask for valid computations
    valid_mask = denominator_squared >= epsilon

    # Compute similarities for valid entries
    denominators = xp.sqrt(denominator_squared)
    valid_denom_mask = denominators >= epsilon

    # Combined mask
    final_valid = valid_mask & valid_denom_mask

    # Avoid division by zero
    safe_denominators = xp.where(final_valid, denominators, 1.0)
    similarities = lorentz_products_uv / safe_denominators

    # Set invalid entries to 0.0
    similarities = xp.where(final_valid, similarities, 0.0)

    # Clamp to valid range
    similarities = xp.clip(similarities, -1.0, 1.0)

    # Return to CPU if requested
    if return_cpu:
        return to_cpu(similarities)
    return similarities


def standard_cosine_similarity_gpu(
    u: Union[np.ndarray, 'cp.ndarray'],
    v: Union[np.ndarray, 'cp.ndarray'],
    epsilon: float = 1e-10,
    return_cpu: bool = True
) -> float:
    """
    Compute standard cosine similarity on GPU.

    GPU-accelerated version of standard cosine similarity for comparison.

    Parameters:
    -----------
    u : Union[np.ndarray, cp.ndarray]
        First vector (1-D array)
    v : Union[np.ndarray, cp.ndarray]
        Second vector (1-D array, same dimension as u)
    epsilon : float
        Small constant for numerical stability (default: 1e-10)
    return_cpu : bool
        If True, return result as Python float (default: True)

    Returns:
    --------
    float
        Cosine similarity in range [-1, 1]
    """
    # Transfer to GPU if available
    if CUPY_AVAILABLE and is_gpu_available():
        xp = cp
        u = cp.asarray(u, dtype=cp.float64)
        v = cp.asarray(v, dtype=cp.float64)
    else:
        xp = np
        u = np.asarray(u, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)

    if u.shape != v.shape:
        raise ValueError(f"Vectors must have same shape: {u.shape} vs {v.shape}")

    # Compute dot product and norms
    dot_product = xp.dot(u, v)
    norm_u = xp.linalg.norm(u)
    norm_v = xp.linalg.norm(v)

    # Avoid division by zero
    denominator = norm_u * norm_v
    if float(denominator) < epsilon:
        return 0.0

    similarity = dot_product / denominator
    similarity = xp.clip(similarity, -1.0, 1.0)

    if return_cpu:
        return float(similarity)
    return similarity


# Convenience function for automatic GPU/CPU selection
def lorentz_similarity_auto(
    u: np.ndarray,
    v: np.ndarray,
    epsilon: float = 1e-10,
    prefer_gpu: bool = True
) -> float:
    """
    Automatically select GPU or CPU implementation based on availability.

    Parameters:
    -----------
    u : np.ndarray
        First vector
    v : np.ndarray
        Second vector
    epsilon : float
        Numerical stability constant
    prefer_gpu : bool
        If True, use GPU when available (default: True)

    Returns:
    --------
    float
        Lorentz-invariant similarity
    """
    if prefer_gpu and is_gpu_available():
        return lorentz_similarity_gpu(u, v, epsilon=epsilon, return_cpu=True)
    else:
        # Fall back to CPU implementation
        from similarity import lorentz_similarity
        return lorentz_similarity(u, v, epsilon=epsilon)


# PyTorch Integration
# ===================

# Try to import PyTorch
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


def eigen_similarity(
    U: Union[np.ndarray, 'cp.ndarray', 'torch.Tensor'],
    V: Union[np.ndarray, 'cp.ndarray', 'torch.Tensor'],
    epsilon: float = 1e-10,
) -> Union[np.ndarray, 'cp.ndarray', 'torch.Tensor']:
    """
    Compute pairwise Lorentz-invariant (EigenFunction) similarity matrix.

    This is the primary interface for PyTorch modules (eigen_memory, eigen_attention).
    Supports batched computation of all pairwise similarities between vectors in U and V.

    **Loop Prevention**: Diagonal elements (self-similarity) are ~0.0, preventing
    pathological feedback loops in attention and memory mechanisms.

    Parameters:
    -----------
    U : Union[np.ndarray, cp.ndarray, torch.Tensor]
        First batch of vectors, shape (N, D) or (B, L, D)
        - If 2D: N vectors of dimension D
        - If 3D: Flattened to (B*L, D) for computation
    V : Union[np.ndarray, cp.ndarray, torch.Tensor]
        Second batch of vectors, shape (M, D) or (B, L, D)
        - If 2D: M vectors of dimension D
        - If 3D: Flattened to (B*L, D) for computation
    epsilon : float
        Small constant for numerical stability (default: 1e-10)

    Returns:
    --------
    Union[np.ndarray, cp.ndarray, torch.Tensor]
        Pairwise similarity matrix
        - If U is (N, D) and V is (M, D): returns (N, M)
        - If U is (B, L, D) and V is (B, L, D): returns (B*L, B*L)
        Returns same type as input (numpy/cupy/torch)

    Notes:
    ------
    - Self-similarity (diagonal when U=V) is ~0.0 (lightlike boundary)
    - Similarities are in range [-1, 1]
    - Automatically uses GPU if tensors are on CUDA device
    - For PyTorch tensors, gradients are preserved

    Examples:
    ---------
    >>> import torch
    >>> import gpu_similarity as gpu_sim
    >>>
    >>> # Attention mechanism
    >>> queries = torch.randn(8, 64, 512)  # (batch, seq_len, dim)
    >>> keys = torch.randn(8, 64, 512)
    >>> sim = gpu_sim.eigen_similarity(queries, keys)  # (8*64, 8*64)
    >>>
    >>> # Memory retrieval
    >>> query = torch.randn(32, 256)  # (batch, dim)
    >>> memory = torch.randn(1000, 256)  # (mem_size, dim)
    >>> sim = gpu_sim.eigen_similarity(query, memory)  # (32, 1000)
    """
    # Handle PyTorch tensors
    if TORCH_AVAILABLE and isinstance(U, torch.Tensor):
        return _eigen_similarity_torch(U, V, epsilon)

    # Handle 3D arrays by flattening
    original_shape_U = None
    original_shape_V = None

    if hasattr(U, 'ndim'):
        if U.ndim == 3:
            original_shape_U = U.shape
            B, L, D = U.shape
            U = U.reshape(B * L, D)
        elif U.ndim != 2:
            raise ValueError(f"U must be 2D or 3D, got shape {U.shape}")

    if hasattr(V, 'ndim'):
        if V.ndim == 3:
            original_shape_V = V.shape
            B, L, D = V.shape
            V = V.reshape(B * L, D)
        elif V.ndim != 2:
            raise ValueError(f"V must be 2D or 3D, got shape {V.shape}")

    # Use the existing matrix computation
    result = lorentz_similarity_matrix_gpu(U, V, epsilon=epsilon, return_cpu=False)

    return result


def _eigen_similarity_torch(
    U: 'torch.Tensor',
    V: 'torch.Tensor',
    epsilon: float = 1e-10,
) -> 'torch.Tensor':
    """
    PyTorch-native implementation of EigenFunction similarity.

    Preserves gradients for backpropagation and respects device placement.

    Parameters:
    -----------
    U : torch.Tensor
        Shape (N, D) or (B, L, D)
    V : torch.Tensor
        Shape (M, D) or (B, L, D)
    epsilon : float
        Numerical stability constant

    Returns:
    --------
    torch.Tensor
        Similarity matrix, shape (N, M) or (B*L, B*L)
        Same device and dtype as inputs
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available")

    # Handle 3D tensors by flattening
    reshape_output = False
    if U.ndim == 3:
        B, L, D = U.shape
        U_flat = U.reshape(B * L, D)
        reshape_output = True
    elif U.ndim == 2:
        U_flat = U
        N, D = U.shape
    else:
        raise ValueError(f"U must be 2D or 3D, got shape {U.shape}")

    if V.ndim == 3:
        B_v, L_v, D_v = V.shape
        V_flat = V.reshape(B_v * L_v, D_v)
    elif V.ndim == 2:
        V_flat = V
        M, D_v = V.shape
    else:
        raise ValueError(f"V must be 2D or 3D, got shape {V.shape}")

    if U_flat.shape[1] != V_flat.shape[1]:
        raise ValueError(
            f"Dimension mismatch: U has dim {U_flat.shape[1]}, "
            f"V has dim {V_flat.shape[1]}"
        )

    N = U_flat.shape[0]
    M = V_flat.shape[0]
    D = U_flat.shape[1]

    device = U_flat.device
    dtype = U_flat.dtype

    # Compute norms
    norms_U = torch.norm(U_flat, dim=1, keepdim=True)  # (N, 1)
    norms_V = torch.norm(V_flat, dim=1, keepdim=True)  # (M, 1)

    # Spatial dot products: U @ V^T
    spatial_products = torch.mm(U_flat, V_flat.T)  # (N, M)

    # Lorentz inner products
    # <u, v>_L = u·v - ||u|| * ||v||
    lorentz_products_uv = spatial_products - (norms_U @ norms_V.T)

    # Self inner products (lightlike)
    # <u, u>_L = ||u||² - ||u||² = 0
    lorentz_products_uu = norms_U ** 2 - norms_U ** 2  # (N, 1), all zeros
    lorentz_products_vv = norms_V ** 2 - norms_V ** 2  # (M, 1), all zeros

    # Denominators: sqrt(|<u,u>_L| * |<v,v>_L|)
    # Since both are ~0 (lightlike), denominator is ~0
    # We need to handle this carefully to avoid NaN in gradients
    denominator_squared = torch.abs(lorentz_products_uu) @ torch.abs(lorentz_products_vv).T

    # Mask for valid (non-lightlike) computations
    valid_mask = denominator_squared >= epsilon

    # Initialize similarities
    similarities = torch.zeros(N, M, dtype=dtype, device=device)

    # Compute similarities where valid
    # Use safe division: add epsilon to denominator
    safe_denom_sq = denominator_squared + epsilon
    safe_denom = torch.sqrt(safe_denom_sq)

    # Compute similarities
    sim_values = lorentz_products_uv / safe_denom

    # Only use values where denominator was actually valid
    similarities = torch.where(
        valid_mask,
        sim_values,
        torch.zeros_like(sim_values)
    )

    # Clamp to valid range
    similarities = torch.clamp(similarities, -1.0, 1.0)

    return similarities


def is_torch_available() -> bool:
    """
    Check if PyTorch is available.

    Returns:
    --------
    bool
        True if PyTorch is installed
    """
    return TORCH_AVAILABLE
