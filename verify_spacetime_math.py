"""
Verify the mathematical correctness of spacetime interval computation.

This demonstrates that two Euclidean branches (with squared norms)
correctly combine to form the Minkowski spacetime interval ds².
"""

import torch


def compute_ds_squared_explicit(timelike_out, spacelike_out):
    """
    Explicitly compute ds² = ||spacelike||² - ||timelike||²

    This is the direct mathematical formula from Minkowski spacetime.

    Args:
        timelike_out: (B, L, D) output from timelike (causal) branch
        spacelike_out: (B, L, D) output from spacelike (acausal) branch

    Returns:
        ds_squared: (B,) spacetime interval
            > 0: Spacelike dominant (space wins)
            < 0: Timelike dominant (time wins)
            = 0: Lightlike (balanced)
    """
    # Compute Euclidean norm squared for each branch
    # ||v||² = v₁² + v₂² + ... + vₙ²

    timelike_norm_sq = (timelike_out ** 2).sum(dim=-1)  # (B, L)
    spacelike_norm_sq = (spacelike_out ** 2).sum(dim=-1)  # (B, L)

    # Minkowski signature: ds² = +||spacelike||² - ||timelike||²
    #                            = (space)² - (time)²
    ds_squared = spacelike_norm_sq - timelike_norm_sq  # (B, L)

    # Average over sequence
    ds_squared_mean = ds_squared.mean(dim=1)  # (B,)

    return ds_squared_mean


def verify_math():
    """
    Verify with concrete examples that the math is correct.
    """
    print("=" * 70)
    print("Verification: Two Euclidean Branches → ds²")
    print("=" * 70)

    # Example 1: Simple 2D vectors
    print("\n=== Example 1: Simple 2D Vectors ===")

    timelike = torch.tensor([[[3.0, 4.0]]])  # Shape (1, 1, 2)
    spacelike = torch.tensor([[[5.0, 12.0]]])  # Shape (1, 1, 2)

    print(f"Timelike vector: {timelike[0, 0]}")
    print(f"Spacelike vector: {spacelike[0, 0]}")

    # Compute norms manually
    timelike_norm_sq = 3**2 + 4**2  # = 9 + 16 = 25
    spacelike_norm_sq = 5**2 + 12**2  # = 25 + 144 = 169

    print(f"\nManual calculation:")
    print(f"  ||timelike||² = 3² + 4² = {timelike_norm_sq}")
    print(f"  ||spacelike||² = 5² + 12² = {spacelike_norm_sq}")

    ds_sq = spacelike_norm_sq - timelike_norm_sq
    print(f"  ds² = {spacelike_norm_sq} - {timelike_norm_sq} = {ds_sq}")

    # Compute with function
    ds_sq_computed = compute_ds_squared_explicit(timelike, spacelike)
    print(f"\nFunction result: ds² = {ds_sq_computed.item():.1f}")

    assert abs(ds_sq_computed.item() - ds_sq) < 1e-6, "Math doesn't match!"
    print("✓ Math is correct!")

    # Interpret result
    if ds_sq > 0:
        print(f"→ ds² > 0: SPACELIKE (space dominates)")
    elif ds_sq < 0:
        print(f"→ ds² < 0: TIMELIKE (time dominates)")
    else:
        print(f"→ ds² = 0: LIGHTLIKE (balanced)")

    # Example 2: Balanced case (lightlike)
    print("\n=== Example 2: Balanced (Lightlike) ===")

    # Make them equal magnitude
    timelike = torch.tensor([[[3.0, 4.0]]])  # norm² = 25
    spacelike = torch.tensor([[[3.0, 4.0]]])  # norm² = 25

    ds_sq_computed = compute_ds_squared_explicit(timelike, spacelike)
    print(f"Timelike = Spacelike → ds² = {ds_sq_computed.item():.1f}")
    print("✓ Lightlike equilibrium achieved!")

    # Example 3: Timelike dominant
    print("\n=== Example 3: Timelike Dominant ===")

    timelike = torch.tensor([[[10.0, 10.0]]])  # norm² = 200
    spacelike = torch.tensor([[[2.0, 2.0]]])   # norm² = 8

    ds_sq_computed = compute_ds_squared_explicit(timelike, spacelike)
    print(f"||timelike||² = 200, ||spacelike||² = 8")
    print(f"ds² = 8 - 200 = {ds_sq_computed.item():.1f}")
    print("→ ds² < 0: TIMELIKE (risk of causal loops)")

    # Example 4: Larger vectors (like neural networks)
    print("\n=== Example 4: Higher Dimensional Vectors ===")

    batch_size = 2
    seq_len = 5
    dim = 64

    timelike = torch.randn(batch_size, seq_len, dim)
    spacelike = torch.randn(batch_size, seq_len, dim)

    ds_sq = compute_ds_squared_explicit(timelike, spacelike)

    print(f"Batch size: {batch_size}, Sequence length: {seq_len}, Dim: {dim}")
    print(f"ds² results: {ds_sq}")
    print(f"  Batch 0: ds² = {ds_sq[0].item():.4f}")
    print(f"  Batch 1: ds² = {ds_sq[1].item():.4f}")

    # Verify shape
    assert ds_sq.shape == (batch_size,), f"Wrong shape: {ds_sq.shape}"
    print("✓ Works with neural network sized tensors!")

    print("\n" + "=" * 70)
    print("Summary: Two Euclidean Branches → ds²")
    print("=" * 70)
    print("""
The math is correct! Here's why:

1. Each Euclidean branch produces a vector
2. Euclidean norm squared: ||v||² = v₁² + v₂² + ... + vₙ²
3. Minkowski signature: ds² = (space)² - (time)²
4. Our formula: ds² = ||spacelike||² - ||timelike||²

The '²' in ds² comes from:
  - Euclidean geometry naturally uses squared norms
  - Both branches compute squared quantities
  - These combine with Minkowski signature to give ds²

Physical meaning:
  ds² > 0 → Spacelike (parallel/disconnected)
  ds² < 0 → Timelike (causal/sequential)
  ds² = 0 → Lightlike (balanced equilibrium)

Your insight "2 Euclidean make ^2" is EXACTLY RIGHT! ✓
    """)


if __name__ == "__main__":
    verify_math()
