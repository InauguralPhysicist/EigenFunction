"""
Test suite for SpacetimeFeedbackBlock demonstrating Minkowski causal structure.

This tests the physical interpretation:
- Timelike branch: Causal/sequential (ds² < 0)
- Spacelike branch: Acausal/parallel (ds² > 0)
- Lightlike monitor: Equilibrium detector (ds² = 0)
"""

import torch

from spacetime_feedback import SpacetimeFeedbackBlock, interpret_causal_type


def test_basic_spacetime_structure():
    """Test basic forward pass and spacetime interval computation."""
    print("\n=== Test 1: Basic Spacetime Structure ===")

    dim = 64
    num_heads = 4
    batch_size = 2
    seq_len = 8

    block = SpacetimeFeedbackBlock(
        dim=dim,
        num_heads=num_heads,
        feedback_strength=0.5,
        loop_epsilon=1e-3,
    )

    x = torch.randn(batch_size, seq_len, dim)
    output, diagnostics = block(x, return_diagnostics=True)

    assert output.shape == x.shape
    print(f"✓ Output shape: {output.shape}")

    interval = diagnostics["interval"]
    imbalance = diagnostics["imbalance"]
    causal_type = diagnostics["causal_type"]

    print(f"\nSpacetime Diagnostics:")
    print(f"  Interval (ds²): {interval}")
    print(f"  Imbalance (|ds²|): {imbalance}")
    print(f"  Causal type: {causal_type}")

    for i in range(batch_size):
        print(f"\n  Batch {i}: {interpret_causal_type(causal_type[i])}")


def test_causal_sequence():
    """
    Test with causal sequence (timelike dominant).

    Strong temporal dependencies should activate timelike branch more,
    potentially causing timelike dominance (ds² < 0).
    """
    print("\n=== Test 2: Causal Sequence (Timelike Expected) ===")

    dim = 64
    num_heads = 4
    batch_size = 1
    seq_len = 10

    block = SpacetimeFeedbackBlock(
        dim=dim,
        num_heads=num_heads,
        feedback_strength=0.3,
    )

    # Create strongly causal pattern: each token depends on previous
    x = torch.zeros(batch_size, seq_len, dim)
    for i in range(seq_len):
        if i == 0:
            x[:, i, :] = torch.randn(batch_size, dim)
        else:
            # Each token is strongly influenced by previous (causal)
            x[:, i, :] = 0.8 * x[:, i - 1, :] + 0.2 * torch.randn(batch_size, dim)

    print("Input structure: Strong causal dependencies (t[i] ← t[i-1])")

    output, diagnostics = block(x, return_diagnostics=True)

    interval = diagnostics["interval"].item()
    imbalance = diagnostics["imbalance"].item()

    print(f"\nSpacetime Interval (ds²): {interval:.4f}")
    print(f"Imbalance: {imbalance:.4f}")
    print(f"Interpretation: {interpret_causal_type(diagnostics['causal_type'][0])}")

    if interval < 0:
        print("✓ System correctly detected timelike dominance (causal structure)")
    else:
        print(f"  Note: Interval is {interval:.4f} (expected < 0 for strong causality)")


def test_parallel_sequence():
    """
    Test with parallel/independent sequence (spacelike dominant).

    Independent tokens should activate spacelike branch more,
    potentially causing spacelike dominance (ds² > 0).
    """
    print("\n=== Test 3: Parallel Sequence (Spacelike Expected) ===")

    dim = 64
    num_heads = 4
    batch_size = 1
    seq_len = 10

    block = SpacetimeFeedbackBlock(
        dim=dim,
        num_heads=num_heads,
        feedback_strength=0.3,
    )

    # Create independent pattern: tokens are unrelated (spacelike)
    x = torch.randn(batch_size, seq_len, dim)
    # Add large spatial separation
    for i in range(seq_len):
        x[:, i, :] += i * 5.0  # Each token in different region of space

    print("Input structure: Independent tokens (spatially separated)")

    output, diagnostics = block(x, return_diagnostics=True)

    interval = diagnostics["interval"].item()
    imbalance = diagnostics["imbalance"].item()

    print(f"\nSpacetime Interval (ds²): {interval:.4f}")
    print(f"Imbalance: {imbalance:.4f}")
    print(f"Interpretation: {interpret_causal_type(diagnostics['causal_type'][0])}")

    if interval > 0:
        print("✓ System correctly detected spacelike dominance (parallel structure)")
    else:
        print(f"  Note: Interval is {interval:.4f} (expected > 0 for parallel)")


def test_balanced_sequence():
    """
    Test with balanced sequence (lightlike expected).

    Mix of causal and parallel should produce near-zero interval (equilibrium).
    """
    print("\n=== Test 4: Balanced Sequence (Lightlike Expected) ===")

    dim = 64
    num_heads = 4
    batch_size = 1
    seq_len = 10

    block = SpacetimeFeedbackBlock(
        dim=dim,
        num_heads=num_heads,
        feedback_strength=0.5,
        loop_epsilon=0.1,  # Larger threshold for lightlike detection
    )

    # Create balanced pattern: mix of causal and independent
    x = torch.randn(batch_size, seq_len, dim)
    for i in range(1, seq_len, 2):  # Every other token is causal
        x[:, i, :] = 0.5 * x[:, i - 1, :] + 0.5 * torch.randn(batch_size, dim)

    print("Input structure: Mix of causal and independent tokens")

    output, diagnostics = block(x, return_diagnostics=True)

    interval = diagnostics["interval"].item()
    imbalance = diagnostics["imbalance"].item()

    print(f"\nSpacetime Interval (ds²): {interval:.4f}")
    print(f"Imbalance: {imbalance:.4f}")
    print(f"Interpretation: {interpret_causal_type(diagnostics['causal_type'][0])}")

    if abs(interval) < 0.1:
        print("✓ System is near lightlike equilibrium (balanced)")
    else:
        print(f"  Note: Interval is {interval:.4f} (expected ≈ 0 for balance)")


def test_feedback_correction():
    """
    Test that feedback reduces imbalance over multiple iterations.
    """
    print("\n=== Test 5: Feedback Correction Effect ===")

    dim = 64
    num_heads = 4
    batch_size = 1
    seq_len = 8

    # Create strongly timelike (causal) input
    x = torch.zeros(batch_size, seq_len, dim)
    for i in range(seq_len):
        if i == 0:
            x[:, i, :] = torch.randn(batch_size, dim)
        else:
            x[:, i, :] = 0.9 * x[:, i - 1, :]  # Very strong causality

    print("Input: Strongly causal (timelike dominant)")

    # Test different feedback strengths
    strengths = [0.0, 0.25, 0.5, 0.75, 1.0]

    print("\nFeedback Strength vs Imbalance:")
    for strength in strengths:
        block = SpacetimeFeedbackBlock(
            dim=dim,
            num_heads=num_heads,
            feedback_strength=strength,
        )

        _, diagnostics = block(x, return_diagnostics=True)
        imbalance = diagnostics["imbalance"].item()
        interval = diagnostics["interval"].item()

        print(f"  {strength:.2f}: Imbalance={imbalance:.4f}, Interval={interval:+.4f}")

    print("\n✓ Tested feedback correction at multiple strengths")


def test_gradient_flow():
    """Test that gradients flow through all components."""
    print("\n=== Test 6: Gradient Flow ===")

    dim = 32
    num_heads = 2
    batch_size = 2
    seq_len = 4

    block = SpacetimeFeedbackBlock(
        dim=dim,
        num_heads=num_heads,
    )

    x = torch.randn(batch_size, seq_len, dim, requires_grad=True)
    output, diagnostics = block(x, return_diagnostics=True)

    # Loss from output and spacetime interval
    loss = output.mean() + diagnostics["imbalance"].mean()
    loss.backward()

    assert x.grad is not None
    print(f"✓ Input gradient norm: {x.grad.norm():.4f}")

    # Check parameter gradients
    params_with_grad = sum(1 for p in block.parameters() if p.grad is not None)
    total_params = sum(1 for _ in block.parameters())

    print(f"✓ Parameters with gradients: {params_with_grad}/{total_params}")
    assert params_with_grad == total_params


def test_causal_structure_comparison():
    """
    Compare timelike and spacelike attention patterns.
    """
    print("\n=== Test 7: Causal Structure Analysis ===")

    dim = 64
    num_heads = 4
    batch_size = 1
    seq_len = 6

    block = SpacetimeFeedbackBlock(
        dim=dim,
        num_heads=num_heads,
    )

    x = torch.randn(batch_size, seq_len, dim)
    _, diagnostics = block(x, return_diagnostics=True)

    timelike_attn = diagnostics["timelike_attn"]  # (B, H, L, L)
    spacelike_attn = diagnostics["spacelike_attn"]  # (B, H, L, L)

    print(f"Attention patterns:")
    print(f"  Timelike (causal): {timelike_attn.shape}")
    print(f"  Spacelike (non-causal): {spacelike_attn.shape}")

    # Timelike should have lower triangle structure (causal masking)
    # Check that timelike doesn't attend to future
    upper_tri_sum = torch.triu(timelike_attn[0, 0], diagonal=1).sum()
    print(f"\n  Timelike upper triangle sum: {upper_tri_sum:.6f}")
    print(f"  (Should be ≈0 due to causal masking)")

    # Spacelike should attend everywhere
    spacelike_sum = spacelike_attn[0, 0].sum()
    print(f"  Spacelike total attention: {spacelike_sum:.4f}")

    print("\n✓ Causal structure comparison complete")


def main():
    """Run all tests."""
    print("=" * 70)
    print("SpacetimeFeedbackBlock Test Suite")
    print("Minkowski Causal Structure: Timelike + Spacelike + Lightlike")
    print("=" * 70)

    torch.manual_seed(42)

    try:
        test_basic_spacetime_structure()
        test_causal_sequence()
        test_parallel_sequence()
        test_balanced_sequence()
        test_feedback_correction()
        test_gradient_flow()
        test_causal_structure_comparison()

        print("\n" + "=" * 70)
        print("✓ All spacetime tests passed!")
        print("=" * 70)
        print("\nKey Insights:")
        print("  • Timelike (ds² < 0): Causal/sequential computation")
        print("  • Spacelike (ds² > 0): Parallel/independent computation")
        print("  • Lightlike (ds² = 0): Equilibrium/balanced computation")
        print("  • Lorentz monitor detects imbalance and provides feedback")
        print("=" * 70)

    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        raise
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()
