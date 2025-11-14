"""
Reasoning Chain Validator

Task: Given a chain of logical steps, detect if it's valid or circular.
This is testable without training - just structural analysis.

Example valid chain:
  A → B → C (linear reasoning)

Example circular chain:
  A → B → C → A (circular reasoning, invalid)

Standard models: Can't reliably detect circularity in long chains
Spacetime feedback: Should detect when reasoning loops back
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import List, Tuple


def create_reasoning_chain(chain_type: str, length: int = 5, dim: int = 64) -> torch.Tensor:
    """Create a reasoning chain embedding."""
    x = torch.randn(1, length, dim)

    if chain_type == "linear":
        # A → B → C → D (valid linear reasoning)
        for i in range(1, length):
            x[:, i, :] = 0.6 * x[:, i - 1, :] + 0.4 * torch.randn(1, dim)

    elif chain_type == "circular":
        # A → B → C → A (circular, invalid)
        for i in range(1, length):
            x[:, i, :] = 0.7 * x[:, i - 1, :] + 0.3 * torch.randn(1, dim)
        # Close the loop: last step connects to first
        x[:, -1, :] = 0.8 * x[:, 0, :] + 0.2 * x[:, -1, :]

    elif chain_type == "branching":
        # A → B → C
        #      ↓
        #      D (valid branching)
        mid = length // 2
        for i in range(1, mid):
            x[:, i, :] = 0.6 * x[:, i - 1, :] + 0.4 * torch.randn(1, dim)
        for i in range(mid, length):
            x[:, i, :] = 0.6 * x[:, mid - 1, :] + 0.4 * torch.randn(1, dim)

    elif chain_type == "contradictory":
        # A → B → ¬B (contradiction)
        for i in range(1, length // 2):
            x[:, i, :] = 0.7 * x[:, i - 1, :] + 0.3 * torch.randn(1, dim)
        for i in range(length // 2, length):
            x[:, i, :] = -0.7 * x[:, i - 1, :]  # Negation

    return x


def validate_chain_with_spacetime(x: torch.Tensor) -> Tuple[bool, dict]:
    """
    Validate reasoning chain using spacetime feedback.

    Valid chains: Converge to equilibrium (ds² → 0)
    Invalid chains: Oscillate or diverge (ds² ≠ 0)
    """
    from spacetime_feedback import SpacetimeFeedbackBlock

    dim = x.shape[-1]
    validator = SpacetimeFeedbackBlock(dim, num_heads=4, feedback_strength=0.5)

    intervals = []
    imbalances = []

    # Process chain through spacetime feedback
    state = x
    for _ in range(3):
        state, diag = validator(state, return_diagnostics=True)
        intervals.append(diag["interval"].mean().item())
        imbalances.append(diag["imbalance"].mean().item())

    # Check convergence
    final_imbalance = imbalances[-1]
    avg_imbalance = sum(imbalances) / len(imbalances)

    # Count oscillations
    oscillations = sum(1 for i in range(1, len(intervals)) if intervals[i] * intervals[i - 1] < 0)

    # Valid if converges and doesn't oscillate much
    is_valid = final_imbalance < 0.2 and oscillations < 2

    return is_valid, {
        "intervals": intervals,
        "imbalances": imbalances,
        "oscillations": oscillations,
        "final_imbalance": final_imbalance,
        "avg_imbalance": avg_imbalance,
    }


def test_chain_validation():
    """Test chain validator on different reasoning patterns."""
    print("=" * 80)
    print("Reasoning Chain Validation")
    print("=" * 80)

    torch.manual_seed(42)
    dim = 64

    test_cases = [
        ("linear", True, "A → B → C → D (valid linear)"),
        ("circular", False, "A → B → C → A (circular, invalid)"),
        ("branching", True, "A → B → C/D (valid branching)"),
        ("contradictory", False, "A → B → ¬B (contradiction)"),
    ]

    results = []

    for chain_type, expected_valid, description in test_cases:
        print(f"\n{'-' * 80}")
        print(f"Chain: {description}")
        print(f"Expected: {'Valid' if expected_valid else 'Invalid'}")

        # Create chain
        chain = create_reasoning_chain(chain_type, length=6, dim=dim)

        # Validate
        is_valid, diagnostics = validate_chain_with_spacetime(chain)

        print(f"Detected: {'Valid' if is_valid else 'Invalid'}")
        print(f"Final imbalance: {diagnostics['final_imbalance']:.4f}")
        print(f"Oscillations: {diagnostics['oscillations']}")
        print(f"ds² trace: ", end="")
        for interval in diagnostics["intervals"]:
            sign = "+" if interval >= 0 else "-"
            print(f"{sign}{abs(interval):.3f} ", end="")
        print()

        correct = is_valid == expected_valid
        results.append(
            {
                "description": description,
                "expected": expected_valid,
                "detected": is_valid,
                "correct": correct,
            }
        )

        status = "✓" if correct else "✗"
        print(f"{status} {'Correct' if correct else 'Wrong'}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    correct = sum(1 for r in results if r["correct"])
    total = len(results)
    print(f"\nAccuracy: {correct}/{total} ({correct/total*100:.1f}%)")

    print("\nResults:")
    for r in results:
        status = "✓" if r["correct"] else "✗"
        expected_str = "Valid" if r["expected"] else "Invalid"
        detected_str = "Valid" if r["detected"] else "Invalid"
        print(f"  {status} {r['description']}")
        print(f"     Expected: {expected_str}, Got: {detected_str}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    test_chain_validation()
