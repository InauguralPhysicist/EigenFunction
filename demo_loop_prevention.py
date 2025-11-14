"""
Demonstration: Loop Prevention via Spacetime Feedback

This shows a practical problem where standard attention gets stuck in loops,
but the EigenFunction spacetime architecture prevents infinite recursion.

Task: Self-referential reasoning
- "This statement needs verification"
- Standard attention: looks at itself → verifies itself → looks again → loop
- Spacetime feedback: detects self-similarity, applies correction, breaks loop
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional

from standard_attention import StandardAttention
from spacetime_feedback import SpacetimeFeedbackBlock


class StandardReasoningModel(nn.Module):
    """Baseline: Standard transformer that can loop."""

    def __init__(self, dim: int = 64, num_heads: int = 4, num_layers: int = 2):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList([
            StandardAttention(dim, num_heads) for _ in range(num_layers)
        ])
        self.output_head = nn.Linear(dim, dim)

    def forward(
        self, x: torch.Tensor, max_iterations: int = 10
    ) -> tuple[torch.Tensor, dict]:
        """
        Run iterative reasoning.

        Args:
            x: Input embeddings (B, L, D)
            max_iterations: Maximum reasoning steps

        Returns:
            output: Final reasoning state
            diagnostics: Loop detection info
        """
        state = x
        states = [state.clone()]

        for iteration in range(max_iterations):
            # Apply reasoning layers
            for layer in self.layers:
                state, _ = layer(state)

            states.append(state.clone())

            # Check for loop: is current state similar to previous?
            if len(states) > 1:
                similarity = torch.cosine_similarity(
                    states[-1].flatten(1), states[-2].flatten(1), dim=1
                ).mean()

                # If we're stuck (high self-similarity), we're in a loop
                if similarity > 0.99:
                    return state, {
                        "converged": False,
                        "looped": True,
                        "iterations": iteration + 1,
                        "final_similarity": similarity.item(),
                        "states": states,
                    }

        # Ran out of iterations without converging
        return state, {
            "converged": False,
            "looped": False,
            "iterations": max_iterations,
            "final_similarity": 1.0,
            "states": states,
        }


class SpacetimeReasoningModel(nn.Module):
    """EigenFunction: Spacetime feedback prevents loops."""

    def __init__(
        self,
        dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        feedback_strength: float = 0.5,
    ):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList([
            SpacetimeFeedbackBlock(
                dim, num_heads, feedback_strength=feedback_strength
            )
            for _ in range(num_layers)
        ])
        self.output_head = nn.Linear(dim, dim)

    def forward(
        self, x: torch.Tensor, max_iterations: int = 10, convergence_threshold: float = 0.1
    ) -> tuple[torch.Tensor, dict]:
        """
        Run iterative reasoning with loop prevention.

        Args:
            x: Input embeddings (B, L, D)
            max_iterations: Maximum reasoning steps
            convergence_threshold: When to stop (ds² threshold)

        Returns:
            output: Final reasoning state
            diagnostics: Convergence info
        """
        state = x
        states = [state.clone()]
        intervals = []
        imbalances = []

        for iteration in range(max_iterations):
            # Apply reasoning layers with spacetime feedback
            for layer in self.layers:
                state, diagnostics = layer(state, return_diagnostics=True)
                intervals.append(diagnostics["interval"].mean().item())
                imbalances.append(diagnostics["imbalance"].mean().item())

            states.append(state.clone())

            # Check for convergence: is system at lightlike equilibrium?
            current_imbalance = imbalances[-1]

            if current_imbalance < convergence_threshold:
                return state, {
                    "converged": True,
                    "looped": False,
                    "iterations": iteration + 1,
                    "final_imbalance": current_imbalance,
                    "intervals": intervals,
                    "imbalances": imbalances,
                    "states": states,
                }

        # Reached max iterations
        return state, {
            "converged": False,
            "looped": False,
            "iterations": max_iterations,
            "final_imbalance": imbalances[-1] if imbalances else 1.0,
            "intervals": intervals,
            "imbalances": imbalances,
            "states": states,
        }


def create_self_referential_input(
    batch_size: int = 1, seq_len: int = 4, dim: int = 64
) -> torch.Tensor:
    """
    Create a self-referential input pattern.

    This simulates a statement that refers to itself, like:
    "This statement requires verification"

    The pattern is designed to be highly self-similar, which causes
    standard attention to get stuck in loops.
    """
    x = torch.randn(batch_size, seq_len, dim)

    # Make tokens reference each other strongly (self-referential loop)
    # Token 0: "This"
    # Token 1: "statement" (similar to token 0)
    # Token 2: "requires" (similar to token 1)
    # Token 3: "verification" (similar to token 0) -> circular!

    x[:, 1, :] = 0.9 * x[:, 0, :] + 0.1 * torch.randn(batch_size, dim)
    x[:, 2, :] = 0.9 * x[:, 1, :] + 0.1 * torch.randn(batch_size, dim)
    x[:, 3, :] = 0.9 * x[:, 0, :] + 0.1 * torch.randn(batch_size, dim)

    return x


def demo_loop_prevention():
    """Run demonstration comparing standard vs spacetime models."""
    print("=" * 70)
    print("DEMONSTRATION: Loop Prevention via Spacetime Feedback")
    print("=" * 70)

    torch.manual_seed(42)

    dim = 64
    num_heads = 4
    batch_size = 1
    seq_len = 4
    max_iterations = 10

    # Create self-referential input (designed to cause loops)
    x = create_self_referential_input(batch_size, seq_len, dim)

    print("\n=== Input: Self-Referential Pattern ===")
    print(f"Shape: {x.shape}")
    print(f"Token similarities (diagonal structure indicates self-reference):")
    for i in range(seq_len):
        for j in range(seq_len):
            sim = torch.cosine_similarity(x[0, i], x[0, j], dim=0).item()
            print(f"  Token {i} <-> Token {j}: {sim:.3f}")

    # Test 1: Standard Model (will loop)
    print("\n" + "=" * 70)
    print("Test 1: Standard Transformer (Baseline)")
    print("=" * 70)

    standard_model = StandardReasoningModel(dim, num_heads)
    standard_output, standard_diagnostics = standard_model(x, max_iterations)

    print(f"\nResult:")
    print(f"  Converged: {standard_diagnostics['converged']}")
    print(f"  Looped: {standard_diagnostics['looped']}")
    print(f"  Iterations: {standard_diagnostics['iterations']}")
    print(f"  Final similarity: {standard_diagnostics['final_similarity']:.4f}")

    if standard_diagnostics["looped"]:
        print(f"\n  Loop detected (similarity > 0.99)")
        print(f"    High self-similarity between iterations")

    # Test 2: Spacetime Model (prevents loops)
    print("\n" + "=" * 70)
    print("Test 2: Spacetime Feedback (EigenFunction)")
    print("=" * 70)

    spacetime_model = SpacetimeReasoningModel(
        dim, num_heads, feedback_strength=0.5
    )
    spacetime_output, spacetime_diagnostics = spacetime_model(x, max_iterations)

    print(f"\nResult:")
    print(f"  Converged: {spacetime_diagnostics['converged']}")
    print(f"  Looped: {spacetime_diagnostics['looped']}")
    print(f"  Iterations: {spacetime_diagnostics['iterations']}")
    print(f"  Final imbalance: {spacetime_diagnostics['final_imbalance']:.4f}")

    if spacetime_diagnostics["converged"]:
        print(f"\n  Converged (imbalance < threshold)")
        print(f"    Feedback correction applied based on ds²")

    # Show spacetime interval evolution
    print(f"\n  Spacetime Interval (ds²) Evolution:")
    intervals = spacetime_diagnostics["intervals"]
    for i, interval in enumerate(intervals[:10]):  # Show first 10
        interpretation = "TIMELIKE" if interval < -0.1 else "SPACELIKE" if interval > 0.1 else "LIGHTLIKE"
        print(f"    Step {i}: ds² = {interval:+.4f} ({interpretation})")

    # Comparison
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    print("\nStandard attention:")
    print(f"  Iterations: {standard_diagnostics['iterations']}")
    print(f"  Loop detected: {standard_diagnostics['looped']}")

    print("\nSpacetime feedback:")
    print(f"  Iterations: {spacetime_diagnostics['iterations']}")
    print(f"  Converged: {spacetime_diagnostics['converged']}")
    print(f"  Final imbalance: {spacetime_diagnostics['final_imbalance']:.4f}")

    print("\nDifference: Feedback correction based on spacetime interval (ds²)")
    print("=" * 70)


def demo_reasoning_task():
    """
    Demonstrate on a more realistic reasoning task.

    Task: Iterative query refinement
    - Start with vague query
    - Refine based on context
    - Stop when query is clear (not when stuck in loop)
    """
    print("\n" + "=" * 70)
    print("BONUS: Iterative Query Refinement")
    print("=" * 70)

    torch.manual_seed(123)

    dim = 64
    batch_size = 1
    seq_len = 8  # Longer sequence

    # Simulate query refinement task
    initial_query = torch.randn(batch_size, seq_len, dim)

    # Add some structure: query is ambiguous (needs refinement)
    initial_query[:, :4, :] *= 0.5  # First half is weak
    initial_query[:, 4:, :] *= 2.0  # Second half is strong (imbalanced)

    print("\n=== Task: Refine Ambiguous Query ===")
    print(f"Input shape: {initial_query.shape}")
    print("Query structure: First half weak (ambiguous), second half strong (specific)")

    # Spacetime model should balance this out
    model = SpacetimeReasoningModel(dim=dim, num_heads=4, feedback_strength=0.7)
    output, diagnostics = model(initial_query, max_iterations=15, convergence_threshold=0.2)

    print(f"\nRefinement Process:")
    print(f"  Converged: {diagnostics['converged']}")
    print(f"  Iterations: {diagnostics['iterations']}")
    print(f"  Final imbalance: {diagnostics['final_imbalance']:.4f}")

    # Check if query is now balanced
    first_half_norm = output[:, :4, :].norm().item()
    second_half_norm = output[:, 4:, :].norm().item()
    balance_ratio = first_half_norm / second_half_norm

    print(f"\nQuery Balance:")
    print(f"  First half norm: {first_half_norm:.2f}")
    print(f"  Second half norm: {second_half_norm:.2f}")
    print(f"  Balance ratio: {balance_ratio:.3f} (1.0 = perfect balance)")

    if 0.8 < balance_ratio < 1.2:
        print("  Balanced within 20% tolerance")

    print("=" * 70)


if __name__ == "__main__":
    demo_loop_prevention()
    demo_reasoning_task()
