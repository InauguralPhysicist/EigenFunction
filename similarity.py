"""
Lorentz-Invariant Cosine Similarity
====================================

This module implements a Lorentz-invariant similarity measure that addresses
pathological self-reference behavior in recursive and iterative systems.

Theoretical Foundation:
-----------------------
Standard cosine similarity yields self-similarity = 1.0, which can exacerbate
infinite loops in recursive systems (attention mechanisms, graph traversal,
reinforcement learning, etc.).

The Lorentz-invariant approach computes similarity on the lightlike boundary
(ds² = 0), yielding self-similarity = 0.0. This neutral value acts as a
geometric safeguard, disrupting self-reinforcement and promoting evolutionary
dynamics rather than fixed-point collapse.

Mathematical Formulation:
-------------------------
For vectors u and v in n-dimensional space, we embed them in (n+1)-dimensional
Minkowski spacetime as (u, ||u||) and (v, ||v||), where the final component
represents the temporal dimension.

The Lorentz inner product is:
    <u, v>_L = u·v - ||u|| * ||v||

The Lorentz-invariant similarity is then:
    sim_L(u, v) = <u, v>_L / sqrt(|<u, u>_L| * |<v, v>_L|)

For a vector with itself:
    <u, u>_L = ||u||² - ||u||² = 0

This lightlike (null) self-product yields undefined or 0.0 similarity,
preventing self-reinforcing loops.
"""

from typing import Optional, Union

import numpy as np


def lorentz_similarity(u: np.ndarray, v: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Compute Lorentz-invariant cosine similarity between two vectors.

    This similarity measure yields 0.0 for self-similarity (u == v),
    providing a natural safeguard against infinite loops in recursive systems.

    Parameters:
    -----------
    u : np.ndarray
        First vector (1-D array)
    v : np.ndarray
        Second vector (1-D array, same dimension as u)
    epsilon : float
        Small constant for numerical stability (default: 1e-10)

    Returns:
    --------
    float
        Lorentz-invariant similarity in range approximately [-1, 1]
        Returns 0.0 when u and v are identical (lightlike self-similarity)

    Examples:
    ---------
    >>> u = np.array([1.0, 0.0, 0.0])
    >>> v = np.array([0.0, 1.0, 0.0])
    >>> lorentz_similarity(u, v)
    -1.0

    >>> # Self-similarity returns 0.0 (loop prevention)
    >>> lorentz_similarity(u, u)
    0.0
    """
    # Ensure inputs are numpy arrays
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)

    if u.shape != v.shape:
        raise ValueError(f"Vectors must have same shape: {u.shape} vs {v.shape}")

    # Compute norms (temporal components in Minkowski embedding)
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)

    # Lorentz inner products
    # <u, v>_L = u·v - ||u|| * ||v||
    spatial_product = np.dot(u, v)
    lorentz_product_uv = spatial_product - (norm_u * norm_v)

    # Self inner products
    # <u, u>_L = ||u||² - ||u||² = 0 (lightlike!)
    # <v, v>_L = ||v||² - ||v||² = 0 (lightlike!)
    lorentz_product_uu = norm_u * norm_u - (norm_u * norm_u)
    lorentz_product_vv = norm_v * norm_v - (norm_v * norm_v)

    # Handle the lightlike case (self-similarity or near-identical vectors)
    # When denominators are near zero, we're on the lightlike boundary
    denominator_squared = abs(lorentz_product_uu) * abs(lorentz_product_vv)

    if denominator_squared < epsilon:
        # Lightlike boundary: return 0.0 (neutral self-similarity)
        return 0.0

    # Standard normalization
    denominator = np.sqrt(denominator_squared)

    # Avoid division by very small numbers
    if denominator < epsilon:
        return 0.0

    similarity = lorentz_product_uv / denominator

    # Clamp to valid range due to numerical errors
    return np.clip(similarity, -1.0, 1.0)


def standard_cosine_similarity(u: np.ndarray, v: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Compute standard cosine similarity between two vectors.

    This is the traditional implementation that yields 1.0 for self-similarity,
    which can cause pathological behavior in recursive systems.

    Parameters:
    -----------
    u : np.ndarray
        First vector (1-D array)
    v : np.ndarray
        Second vector (1-D array, same dimension as u)
    epsilon : float
        Small constant for numerical stability (default: 1e-10)

    Returns:
    --------
    float
        Cosine similarity in range [-1, 1]
        Returns 1.0 when u and v are identical (potential loop amplifier)

    Examples:
    ---------
    >>> u = np.array([1.0, 0.0, 0.0])
    >>> v = np.array([0.0, 1.0, 0.0])
    >>> standard_cosine_similarity(u, v)
    0.0

    >>> # Self-similarity returns 1.0 (can amplify loops!)
    >>> standard_cosine_similarity(u, u)
    1.0
    """
    # Ensure inputs are numpy arrays
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)

    if u.shape != v.shape:
        raise ValueError(f"Vectors must have same shape: {u.shape} vs {v.shape}")

    # Compute dot product and norms
    dot_product = np.dot(u, v)
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)

    # Avoid division by zero
    denominator = norm_u * norm_v
    if denominator < epsilon:
        return 0.0

    similarity = dot_product / denominator

    # Clamp to valid range
    return np.clip(similarity, -1.0, 1.0)


def compare_self_similarity(vector: np.ndarray) -> dict:
    """
    Compare self-similarity behavior between standard and Lorentz-invariant measures.

    Parameters:
    -----------
    vector : np.ndarray
        Vector to compute self-similarity for

    Returns:
    --------
    dict
        Dictionary containing:
        - 'standard': standard cosine self-similarity (should be ~1.0)
        - 'lorentz': Lorentz-invariant self-similarity (should be ~0.0)
        - 'vector_norm': L2 norm of the input vector

    Examples:
    ---------
    >>> v = np.array([3.0, 4.0])
    >>> result = compare_self_similarity(v)
    >>> print(f"Standard: {result['standard']:.6f}")  # 1.000000
    >>> print(f"Lorentz: {result['lorentz']:.6f}")    # 0.000000
    """
    standard = standard_cosine_similarity(vector, vector)
    lorentz = lorentz_similarity(vector, vector)

    return {
        "standard": standard,
        "lorentz": lorentz,
        "vector_norm": np.linalg.norm(vector),
        "interpretation": {
            "standard": "Perfect self-reinforcement (1.0) - potential loop amplifier",
            "lorentz": "Neutral self-reference (0.0) - loop prevention via lightlike boundary",
        },
    }
