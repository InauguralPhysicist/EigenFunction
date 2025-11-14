"""
Benchmark Suite: Loop-Prone Reasoning Tasks

Tests where standard attention gets stuck in loops,
and spacetime feedback should handle better.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Callable
import time


class BenchmarkTask:
    """Base class for a benchmark task."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def generate_input(self, batch_size: int, seq_len: int, dim: int) -> torch.Tensor:
        """Generate input for this task."""
        raise NotImplementedError

    def check_success(self, output: torch.Tensor, diagnostics: dict) -> bool:
        """Check if output is successful (not looped)."""
        raise NotImplementedError

    def get_metrics(self, output: torch.Tensor, diagnostics: dict) -> dict:
        """Get task-specific metrics."""
        raise NotImplementedError


class SelfReferenceTask(BenchmarkTask):
    """Task with self-referential patterns (A refers to A)."""

    def __init__(self):
        super().__init__(
            name="Self-Reference",
            description="Tokens that refer to themselves (A→A loop)",
        )

    def generate_input(self, batch_size: int, seq_len: int, dim: int) -> torch.Tensor:
        x = torch.randn(batch_size, seq_len, dim)
        # Make each token similar to itself from previous position
        for i in range(1, seq_len):
            x[:, i, :] = 0.9 * x[:, i - 1, :] + 0.1 * torch.randn(batch_size, dim)
        return x

    def check_success(self, output: torch.Tensor, diagnostics: dict) -> bool:
        # Success if converged or didn't loop
        if "converged" in diagnostics:
            return diagnostics["converged"]
        else:
            return not diagnostics.get("looped", False)

    def get_metrics(self, output: torch.Tensor, diagnostics: dict) -> dict:
        return {
            "iterations": diagnostics.get("iterations", 0),
            "looped": diagnostics.get("looped", False),
            "final_value": diagnostics.get("final_imbalance", diagnostics.get("final_similarity", 0)),
        }


class CircularDependencyTask(BenchmarkTask):
    """Task with circular dependencies (A→B→C→A)."""

    def __init__(self):
        super().__init__(
            name="Circular Dependency",
            description="Circular reference chain (A→B→C→A)",
        )

    def generate_input(self, batch_size: int, seq_len: int, dim: int) -> torch.Tensor:
        assert seq_len >= 3, "Need at least 3 tokens for circular dependency"
        x = torch.randn(batch_size, seq_len, dim)

        # Create circular pattern: 0→1→2→0
        x[:, 1, :] = 0.8 * x[:, 0, :] + 0.2 * torch.randn(batch_size, dim)
        x[:, 2, :] = 0.8 * x[:, 1, :] + 0.2 * torch.randn(batch_size, dim)
        # Close the loop
        x[:, 0, :] = 0.8 * x[:, 2, :] + 0.2 * x[:, 0, :]

        return x

    def check_success(self, output: torch.Tensor, diagnostics: dict) -> bool:
        if "converged" in diagnostics:
            return diagnostics["converged"]
        else:
            return not diagnostics.get("looped", False)

    def get_metrics(self, output: torch.Tensor, diagnostics: dict) -> dict:
        return {
            "iterations": diagnostics.get("iterations", 0),
            "looped": diagnostics.get("looped", False),
            "final_value": diagnostics.get("final_imbalance", diagnostics.get("final_similarity", 0)),
        }


class RecursivePlanningTask(BenchmarkTask):
    """Task requiring planning about planning (meta-level)."""

    def __init__(self):
        super().__init__(
            name="Recursive Planning",
            description="Planning to make a plan (meta-level reasoning)",
        )

    def generate_input(self, batch_size: int, seq_len: int, dim: int) -> torch.Tensor:
        x = torch.randn(batch_size, seq_len, dim)

        # First half: "make a plan"
        # Second half: "to make a plan" (refers back to first half)
        mid = seq_len // 2
        for i in range(mid, seq_len):
            # Second half references first half
            x[:, i, :] = 0.7 * x[:, i - mid, :] + 0.3 * torch.randn(batch_size, dim)

        return x

    def check_success(self, output: torch.Tensor, diagnostics: dict) -> bool:
        if "converged" in diagnostics:
            return diagnostics["converged"]
        else:
            return not diagnostics.get("looped", False)

    def get_metrics(self, output: torch.Tensor, diagnostics: dict) -> dict:
        return {
            "iterations": diagnostics.get("iterations", 0),
            "looped": diagnostics.get("looped", False),
            "final_value": diagnostics.get("final_imbalance", diagnostics.get("final_similarity", 0)),
        }


class FixedPointTask(BenchmarkTask):
    """Task requiring finding a fixed point (f(x) = x)."""

    def __init__(self):
        super().__init__(
            name="Fixed Point",
            description="Find stable state where f(x) = x",
        )

    def generate_input(self, batch_size: int, seq_len: int, dim: int) -> torch.Tensor:
        # Start far from fixed point
        x = torch.randn(batch_size, seq_len, dim) * 5.0
        return x

    def check_success(self, output: torch.Tensor, diagnostics: dict) -> bool:
        # Success if we found a stable state
        if "converged" in diagnostics:
            return diagnostics["converged"]
        else:
            # For standard attention, check if similarity is stable but not too high
            final_sim = diagnostics.get("final_similarity", 1.0)
            return 0.85 < final_sim < 0.99

    def get_metrics(self, output: torch.Tensor, diagnostics: dict) -> dict:
        return {
            "iterations": diagnostics.get("iterations", 0),
            "looped": diagnostics.get("looped", False),
            "final_value": diagnostics.get("final_imbalance", diagnostics.get("final_similarity", 0)),
        }


class BenchmarkSuite:
    """Collection of benchmark tasks."""

    def __init__(self):
        self.tasks = [
            SelfReferenceTask(),
            CircularDependencyTask(),
            RecursivePlanningTask(),
            FixedPointTask(),
        ]

    def run_task(
        self,
        task: BenchmarkTask,
        model: nn.Module,
        model_name: str,
        batch_size: int = 1,
        seq_len: int = 8,
        dim: int = 64,
        max_iterations: int = 10,
    ) -> dict:
        """Run a single task on a model."""
        # Generate input
        x = task.generate_input(batch_size, seq_len, dim)

        # Time execution
        start_time = time.time()

        # Run model
        output, diagnostics = model(x, max_iterations=max_iterations)

        elapsed_time = time.time() - start_time

        # Check success
        success = task.check_success(output, diagnostics)

        # Get metrics
        metrics = task.get_metrics(output, diagnostics)

        return {
            "task": task.name,
            "model": model_name,
            "success": success,
            "time": elapsed_time,
            **metrics,
        }

    def run_all(
        self,
        models: list[tuple[str, nn.Module]],
        batch_size: int = 1,
        seq_len: int = 8,
        dim: int = 64,
        max_iterations: int = 10,
        seed: int = 42,
    ) -> list[dict]:
        """Run all tasks on all models."""
        torch.manual_seed(seed)

        results = []

        for task in self.tasks:
            for model_name, model in models:
                # Reset seed for fair comparison
                torch.manual_seed(seed)

                result = self.run_task(
                    task,
                    model,
                    model_name,
                    batch_size,
                    seq_len,
                    dim,
                    max_iterations,
                )
                results.append(result)

        return results

    def print_results(self, results: list[dict]):
        """Print results in a readable format."""
        print("\n" + "=" * 80)
        print("BENCHMARK RESULTS")
        print("=" * 80)

        # Group by task
        tasks = {}
        for result in results:
            task_name = result["task"]
            if task_name not in tasks:
                tasks[task_name] = []
            tasks[task_name].append(result)

        # Print each task
        for task_name, task_results in tasks.items():
            print(f"\n{task_name}")
            print("-" * 80)

            for result in task_results:
                success_str = "✓" if result["success"] else "✗"
                print(f"\n  {success_str} {result['model']}")
                print(f"      Iterations: {result['iterations']}")
                print(f"      Time: {result['time']:.4f}s")
                print(f"      Looped: {result.get('looped', 'N/A')}")
                print(f"      Final value: {result['final_value']:.4f}")

        # Summary
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)

        # Count successes per model
        model_stats = {}
        for result in results:
            model_name = result["model"]
            if model_name not in model_stats:
                model_stats[model_name] = {"success": 0, "total": 0}
            model_stats[model_name]["total"] += 1
            if result["success"]:
                model_stats[model_name]["success"] += 1

        for model_name, stats in model_stats.items():
            success_rate = stats["success"] / stats["total"] * 100
            print(f"\n{model_name}")
            print(f"  Success rate: {stats['success']}/{stats['total']} ({success_rate:.1f}%)")

        print("\n" + "=" * 80)


def run_benchmarks():
    """Run benchmarks comparing standard vs spacetime models."""
    from demo_loop_prevention import StandardReasoningModel, SpacetimeReasoningModel

    print("=" * 80)
    print("Loop-Prone Reasoning Benchmarks")
    print("=" * 80)
    print("\nComparing:")
    print("  1. Standard Attention (baseline)")
    print("  2. Spacetime Feedback (EigenFunction)")

    dim = 64
    num_heads = 4

    # Create models
    standard = StandardReasoningModel(dim=dim, num_heads=num_heads)
    spacetime = SpacetimeReasoningModel(
        dim=dim, num_heads=num_heads, feedback_strength=0.5
    )

    models = [
        ("Standard", standard),
        ("Spacetime", spacetime),
    ]

    # Run benchmarks
    suite = BenchmarkSuite()
    results = suite.run_all(
        models,
        batch_size=1,
        seq_len=8,
        dim=dim,
        max_iterations=10,
    )

    # Print results
    suite.print_results(results)


if __name__ == "__main__":
    run_benchmarks()
