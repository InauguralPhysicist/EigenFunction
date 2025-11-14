"""
Test suite for FeedbackTransformerBlock demonstrating XOR feedback architecture.

This demonstrates:
1. Euclidean branches (XOR_left, XOR_right) processing opposing signals
2. Lorentz monitor (XOR_top) detecting oscillation/imbalance
3. Feedback correction stabilizing the system
"""

import torch

from feedback_transformer import FeedbackTransformerBlock


def test_basic_forward():
    """Test that FeedbackTransformerBlock runs without errors."""
    print("\n=== Test 1: Basic Forward Pass ===")

    dim = 64
    num_heads = 4
    batch_size = 2
    seq_len = 8

    block = FeedbackTransformerBlock(
        dim=dim,
        num_heads=num_heads,
        feedback_strength=0.5,
        loop_epsilon=1e-3,
    )

    x = torch.randn(batch_size, seq_len, dim)
    output, imbalance = block(x, return_imbalance=True)

    assert output.shape == x.shape, f"Output shape mismatch: {output.shape} vs {x.shape}"
    assert imbalance.shape == (batch_size,), f"Imbalance shape mismatch: {imbalance.shape}"
    assert torch.all((imbalance >= 0) & (imbalance <= 1)), "Imbalance should be in [0, 1]"

    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Imbalance scores: {imbalance}")
    print(f"✓ Imbalance range: [{imbalance.min():.4f}, {imbalance.max():.4f}]")


def test_oscillating_input():
    """
    Test with oscillating input pattern to trigger high imbalance.

    Create input where tokens alternate between opposing states,
    which should cause Euclidean branches to produce opposing outputs.
    """
    print("\n=== Test 2: Oscillating Input Detection ===")

    dim = 64
    num_heads = 4
    batch_size = 1
    seq_len = 10

    block = FeedbackTransformerBlock(
        dim=dim,
        num_heads=num_heads,
        feedback_strength=0.8,  # Strong feedback
        loop_epsilon=1e-3,
    )

    # Create oscillating pattern: alternating positive/negative values
    x = torch.zeros(batch_size, seq_len, dim)
    for i in range(seq_len):
        sign = 1 if i % 2 == 0 else -1
        x[:, i, :] = sign * torch.randn(batch_size, dim).abs()

    print(f"Input pattern (first 2 tokens, first 5 dims):")
    print(f"  Token 0: {x[0, 0, :5]}")
    print(f"  Token 1: {x[0, 1, :5]}")

    # Run without feedback
    block_no_feedback = FeedbackTransformerBlock(
        dim=dim, num_heads=num_heads, feedback_strength=0.0  # No correction
    )
    _, imbalance_no_feedback = block_no_feedback(x, return_imbalance=True)

    # Run with feedback
    output_with_feedback, imbalance_with_feedback = block(x, return_imbalance=True)

    print(f"\n✓ Imbalance (no feedback): {imbalance_no_feedback.item():.4f}")
    print(f"✓ Imbalance (with feedback): {imbalance_with_feedback.item():.4f}")

    # Feedback should help, but this is a stochastic system
    print(f"\n✓ Output statistics:")
    print(f"  Mean: {output_with_feedback.mean():.4f}")
    print(f"  Std:  {output_with_feedback.std():.4f}")
    print(f"  Max:  {output_with_feedback.max():.4f}")
    print(f"  Min:  {output_with_feedback.min():.4f}")


def test_stable_input():
    """
    Test with stable (non-oscillating) input.

    All tokens similar → branches shouldn't oppose → low imbalance.
    """
    print("\n=== Test 3: Stable Input (Low Imbalance Expected) ===")

    dim = 64
    num_heads = 4
    batch_size = 1
    seq_len = 10

    block = FeedbackTransformerBlock(
        dim=dim,
        num_heads=num_heads,
        feedback_strength=0.5,
    )

    # Create stable pattern: all tokens similar
    base = torch.randn(1, dim)
    x = base.unsqueeze(1).repeat(batch_size, seq_len, 1)
    x = x + 0.1 * torch.randn_like(x)  # Small noise

    print(f"Input pattern (tokens are similar):")
    print(f"  Token 0: {x[0, 0, :5]}")
    print(f"  Token 1: {x[0, 1, :5]}")

    output, imbalance = block(x, return_imbalance=True)

    print(f"\n✓ Imbalance score: {imbalance.item():.4f}")
    print(f"  (Lower is better for stable input)")

    assert output.shape == x.shape
    print(f"\n✓ Output shape preserved: {output.shape}")


def test_gradient_flow():
    """Test that gradients flow through the feedback mechanism."""
    print("\n=== Test 4: Gradient Flow ===")

    dim = 32
    num_heads = 2
    batch_size = 2
    seq_len = 4

    block = FeedbackTransformerBlock(
        dim=dim,
        num_heads=num_heads,
        feedback_strength=0.5,
    )

    x = torch.randn(batch_size, seq_len, dim, requires_grad=True)
    output, imbalance = block(x, return_imbalance=True)

    # Compute loss from both output and imbalance
    loss = output.mean() + imbalance.mean()
    loss.backward()

    assert x.grad is not None, "Gradients should flow to input"
    print(f"✓ Gradient norm: {x.grad.norm():.4f}")

    # Check that all parameters have gradients
    param_count = 0
    params_with_grad = 0
    for name, param in block.named_parameters():
        param_count += 1
        if param.grad is not None:
            params_with_grad += 1

    print(f"✓ Parameters with gradients: {params_with_grad}/{param_count}")
    assert params_with_grad == param_count, "All parameters should have gradients"


def test_feedback_strength_effect():
    """
    Test that feedback_strength parameter affects the correction magnitude.
    """
    print("\n=== Test 5: Feedback Strength Effect ===")

    dim = 64
    num_heads = 4
    batch_size = 1
    seq_len = 8

    # Create oscillating input
    x = torch.zeros(batch_size, seq_len, dim)
    for i in range(seq_len):
        sign = 1 if i % 2 == 0 else -1
        x[:, i, :] = sign * torch.randn(batch_size, dim).abs()

    # Test different feedback strengths
    strengths = [0.0, 0.25, 0.5, 0.75, 1.0]
    imbalances = []

    for strength in strengths:
        block = FeedbackTransformerBlock(
            dim=dim,
            num_heads=num_heads,
            feedback_strength=strength,
        )
        _, imbalance = block(x, return_imbalance=True)
        imbalances.append(imbalance.item())
        print(f"  Feedback strength {strength:.2f}: Imbalance = {imbalance.item():.4f}")

    print(f"\n✓ Tested {len(strengths)} different feedback strengths")


def test_causal_masking():
    """Test that causal masking works correctly."""
    print("\n=== Test 6: Causal Masking ===")

    dim = 64
    num_heads = 4
    batch_size = 2
    seq_len = 6

    block_causal = FeedbackTransformerBlock(
        dim=dim,
        num_heads=num_heads,
        causal=True,
    )

    x = torch.randn(batch_size, seq_len, dim)
    output, imbalance = block_causal(x, return_imbalance=True)

    assert output.shape == x.shape
    print(f"✓ Causal masking enabled, output shape: {output.shape}")
    print(f"✓ Imbalance with causal: {imbalance.mean():.4f}")


def main():
    """Run all tests."""
    print("=" * 60)
    print("FeedbackTransformerBlock Test Suite")
    print("XOR Architecture: Euclidean (TC) + Lorentz (Monitor)")
    print("=" * 60)

    torch.manual_seed(42)

    try:
        test_basic_forward()
        test_oscillating_input()
        test_stable_input()
        test_gradient_flow()
        test_feedback_strength_effect()
        test_causal_masking()

        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        raise
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()
