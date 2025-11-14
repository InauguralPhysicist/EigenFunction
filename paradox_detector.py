"""
Paradox Detection: Self-Referential Logic Problems

Task: Detect when logical statements create paradoxes (infinite loops).
Current LLMs fail at this - they can't detect circular reasoning reliably.

Examples:
- "This statement is false" (Liar's Paradox)
- "Does the set of all sets that don't contain themselves contain itself?" (Russell)
- "If I say 'I'm lying', am I telling the truth?" (Self-reference)

Standard models: Get confused, give inconsistent answers, can't detect the loop
Spacetime architecture: Should detect the circular reasoning and mark as paradox
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Tuple, List
from enum import Enum


class LogicType(Enum):
    """Types of logical statements."""

    VALID = "valid"  # Can be evaluated to true/false
    PARADOX = "paradox"  # Creates infinite loop (self-contradictory)
    UNKNOWN = "unknown"  # Can't determine


class ParadoxDetector(nn.Module):
    """
    Detect paradoxes using spacetime feedback.

    A paradox creates a loop: evaluating it requires evaluating itself.
    The timelike/spacelike branches will oscillate.
    The lightlike monitor should detect this as high imbalance.
    """

    def __init__(self, dim: int = 64, num_heads: int = 4):
        super().__init__()
        from spacetime_feedback import SpacetimeFeedbackBlock

        self.dim = dim
        self.embedding = nn.Linear(dim, dim)  # Statement embedding

        # Spacetime reasoning layers
        self.reasoner = nn.Sequential(*[
            SpacetimeFeedbackBlock(dim, num_heads, feedback_strength=0.7)
            for _ in range(2)
        ])

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 3),  # 3 classes: valid, paradox, unknown
        )

        # Paradox threshold: if imbalance stays high, it's a paradox
        self.paradox_threshold = 0.3

    def forward(
        self, x: torch.Tensor, max_iterations: int = 5
    ) -> Tuple[LogicType, dict]:
        """
        Evaluate a logical statement.

        Key insight: Paradoxes cause OSCILLATION in ds², not just high imbalance.
        Valid statements converge smoothly, paradoxes oscillate even with feedback.

        Args:
            x: Statement embedding (B, L, D)
            max_iterations: Max reasoning steps

        Returns:
            logic_type: Classification of the statement
            diagnostics: Reasoning trace
        """
        state = self.embedding(x)
        imbalances = []
        intervals = []

        # Try to evaluate the statement
        for i in range(max_iterations):
            for layer in self.reasoner:
                state, diag = layer(state, return_diagnostics=True)
                imbalances.append(diag["imbalance"].mean().item())
                intervals.append(diag["interval"].mean().item())

        # Detect paradox by oscillation pattern, not just magnitude
        # Paradoxes: intervals oscillate between positive/negative (timelike ↔ spacelike)
        # Valid: intervals converge to near-zero (lightlike equilibrium)

        if len(intervals) >= 4:
            # Count sign changes in ds²
            sign_changes = 0
            for i in range(1, len(intervals)):
                if intervals[i] * intervals[i - 1] < 0:  # Sign flip
                    sign_changes += 1

            # Compute variance in imbalance (high variance = oscillating)
            if len(imbalances) > 1:
                mean_imb = sum(imbalances) / len(imbalances)
                variance = sum((x - mean_imb) ** 2 for x in imbalances) / len(imbalances)
            else:
                variance = 0

            # Paradox indicators:
            # 1. High variance (oscillating imbalance)
            # 2. Many sign changes in ds² (flipping between timelike/spacelike)
            # 3. Doesn't converge (imbalance stays significant)

            oscillation_score = sign_changes / len(intervals) + variance
            final_imbalance = sum(imbalances[-3:]) / 3 if len(imbalances) >= 3 else imbalances[-1]

            if oscillation_score > 0.15 or (final_imbalance > 0.15 and variance > 0.002):
                logic_type = LogicType.PARADOX
            else:
                logic_type = LogicType.VALID
        else:
            # Not enough data
            logic_type = LogicType.UNKNOWN

        # Get logits for comparison
        pooled = state.mean(dim=1)  # (B, D)
        logits = self.classifier(pooled)  # (B, 3)

        avg_imbalance = sum(imbalances) / len(imbalances)
        variance = sum((x - avg_imbalance) ** 2 for x in imbalances) / len(imbalances) if len(imbalances) > 1 else 0

        return logic_type, {
            "imbalances": imbalances,
            "intervals": intervals,
            "avg_imbalance": avg_imbalance,
            "variance": variance,
            "oscillation_score": oscillation_score if len(intervals) >= 4 else 0,
            "logits": logits,
            "iterations": len(imbalances),
        }


def create_statement_embedding(
    statement_type: str, batch_size: int = 1, seq_len: int = 4, dim: int = 64
) -> torch.Tensor:
    """
    Create embedding for different statement types.

    We simulate the structure of paradoxes vs valid statements.
    """
    x = torch.randn(batch_size, seq_len, dim)

    if statement_type == "liar_paradox":
        # "This statement is false"
        # Structure: token 0 refers to entire statement (including itself)
        for i in range(seq_len):
            x[:, i, :] = 0.9 * x[:, 0, :] + 0.1 * torch.randn(batch_size, dim)
        # Make first token self-referential
        x[:, 0, :] = 0.8 * x[:, -1, :] + 0.2 * x[:, 0, :]

    elif statement_type == "russell_paradox":
        # "Set of all sets that don't contain themselves"
        # Circular: A ∈ A iff A ∉ A
        x[:, 1, :] = 0.85 * x[:, 0, :] + 0.15 * torch.randn(batch_size, dim)
        x[:, 2, :] = -0.85 * x[:, 1, :]  # Negation
        x[:, 0, :] = 0.85 * x[:, 2, :] + 0.15 * x[:, 0, :]  # Close loop

    elif statement_type == "yesno_paradox":
        # "Is the answer to this question 'no'?"
        # If yes, then no. If no, then yes.
        x[:, 1, :] = 0.9 * x[:, 0, :]
        x[:, 0, :] = -0.9 * x[:, 1, :]  # Negation loop

    elif statement_type == "valid_statement":
        # "2 + 2 = 4" or "The sky is blue"
        # No circular reference
        for i in range(1, seq_len):
            x[:, i, :] = 0.3 * x[:, i - 1, :] + 0.7 * torch.randn(batch_size, dim)

    elif statement_type == "simple_statement":
        # "Cats are mammals"
        # Completely independent tokens
        x = torch.randn(batch_size, seq_len, dim)

    elif statement_type == "asymmetric_equality":
        # "a = b, but b ≠ a" (contradiction - equality is symmetric)
        # Token 0: a, Token 1: =, Token 2: b
        # Token 3: b, Token 4: ≠, Token 5: a
        x[:, 1, :] = 0.8 * (x[:, 0, :] + x[:, 2, :]) / 2  # = relation
        x[:, 4, :] = -0.8 * (x[:, 3, :] + x[:, 5, :]) / 2  # ≠ relation (negated)
        # Create contradiction: same entities, opposite relations
        x[:, 5, :] = 0.9 * x[:, 0, :]  # a appears twice
        x[:, 3, :] = 0.9 * x[:, 2, :]  # b appears twice

    else:
        raise ValueError(f"Unknown statement type: {statement_type}")

    return x


def test_paradox_detection():
    """Test paradox detector on known paradoxes vs valid statements."""
    print("=" * 80)
    print("Paradox Detection Test")
    print("=" * 80)

    torch.manual_seed(42)
    dim = 64

    detector = ParadoxDetector(dim=dim, num_heads=4)

    test_cases = [
        ("liar_paradox", LogicType.PARADOX, "This statement is false"),
        ("russell_paradox", LogicType.PARADOX, "Set that contains itself iff it doesn't"),
        ("yesno_paradox", LogicType.PARADOX, "Is the answer to this question 'no'?"),
        ("asymmetric_equality", LogicType.PARADOX, "a = b, but b ≠ a"),
        ("valid_statement", LogicType.VALID, "2 + 2 = 4"),
        ("simple_statement", LogicType.VALID, "Cats are mammals"),
    ]

    results = []

    for statement_type, expected_type, description in test_cases:
        print(f"\n{'-' * 80}")
        print(f"Statement: {description}")
        print(f"Type: {statement_type}")
        print(f"Expected: {expected_type.value}")

        # Create embedding (use more tokens for asymmetric equality)
        seq_len = 6 if statement_type == "asymmetric_equality" else 4
        x = create_statement_embedding(statement_type, seq_len=seq_len, dim=dim)

        # Detect
        detected_type, diagnostics = detector(x, max_iterations=5)

        print(f"Detected: {detected_type.value}")
        print(f"Average imbalance: {diagnostics['avg_imbalance']:.4f}")
        print(f"Variance: {diagnostics['variance']:.4f}")
        print(f"Oscillation score: {diagnostics['oscillation_score']:.4f}")
        print(f"Iterations: {diagnostics['iterations']}")

        # Show ds² sign changes (key indicator)
        print(f"ds² trace: ", end="")
        for interval in diagnostics["intervals"][:10]:
            sign = "+" if interval >= 0 else "-"
            print(f"{sign}{abs(interval):.3f} ", end="")
        print()

        # Check if correct
        correct = detected_type == expected_type
        results.append({
            "description": description,
            "expected": expected_type,
            "detected": detected_type,
            "correct": correct,
            "avg_imbalance": diagnostics["avg_imbalance"],
            "oscillation_score": diagnostics["oscillation_score"],
        })

        status = "✓" if correct else "✗"
        print(f"{status} {'Correct' if correct else 'Wrong'}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    correct_count = sum(1 for r in results if r["correct"])
    total_count = len(results)

    print(f"\nAccuracy: {correct_count}/{total_count} ({correct_count/total_count*100:.1f}%)")

    print("\nDetailed Results:")
    for r in results:
        status = "✓" if r["correct"] else "✗"
        print(f"  {status} {r['description']}")
        print(f"     Expected: {r['expected'].value}, Got: {r['detected'].value}")
        print(f"     Oscillation: {r['oscillation_score']:.4f}")

    print("\n" + "=" * 80)
    print("How This Works")
    print("=" * 80)
    print("""
Paradoxes create logical loops that cause OSCILLATION:
- "This is false" → if true then false, if false then true → flip-flop
- ds² oscillates between timelike (causal) and spacelike (parallel)

Detection method:
- Count sign changes in ds² (oscillation frequency)
- Measure variance in imbalance (instability)
- Paradoxes oscillate even with feedback correction
- Valid statements converge to lightlike equilibrium (ds² → 0)

Key metric: Oscillation score (sign changes + variance)
            High oscillation (>0.15) = paradox
            Low oscillation (<0.15) = valid statement
    """)
    print("=" * 80)


if __name__ == "__main__":
    test_paradox_detection()
