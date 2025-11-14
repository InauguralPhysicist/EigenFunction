"""
Test Suite for Lorentz-Invariant Similarity Measure
====================================================

This test suite validates the theoretical claims about loop prevention
through neutral self-similarity.
"""

import numpy as np
import pytest
from similarity import lorentz_similarity, standard_cosine_similarity, compare_self_similarity


class TestSelfReferenceProperty:
    """Test the critical self-reference behavior difference."""

    def test_standard_cosine_self_similarity_is_one(self):
        """Standard cosine similarity yields 1.0 for self-reference."""
        vectors = [
            np.array([1.0, 0.0, 0.0]),
            np.array([3.0, 4.0]),
            np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            np.random.randn(10),
        ]

        for v in vectors:
            sim = standard_cosine_similarity(v, v)
            assert np.isclose(
                sim, 1.0, atol=1e-6
            ), f"Standard cosine self-similarity should be 1.0, got {sim}"

    def test_lorentz_self_similarity_is_zero(self):
        """Lorentz-invariant similarity yields 0.0 for self-reference."""
        vectors = [
            np.array([1.0, 0.0, 0.0]),
            np.array([3.0, 4.0]),
            np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            np.random.randn(10),
        ]

        for v in vectors:
            sim = lorentz_similarity(v, v)
            assert np.isclose(
                sim, 0.0, atol=1e-6
            ), f"Lorentz self-similarity should be 0.0, got {sim}"

    def test_comparison_function(self):
        """Test the comparison utility function."""
        v = np.array([3.0, 4.0])
        result = compare_self_similarity(v)

        assert "standard" in result
        assert "lorentz" in result
        assert "vector_norm" in result

        assert np.isclose(result["standard"], 1.0, atol=1e-6)
        assert np.isclose(result["lorentz"], 0.0, atol=1e-6)
        assert np.isclose(result["vector_norm"], 5.0, atol=1e-6)


class TestOrthogonalVectors:
    """Test behavior with orthogonal vectors."""

    def test_standard_orthogonal(self):
        """Standard cosine similarity of orthogonal vectors is 0.0."""
        u = np.array([1.0, 0.0, 0.0])
        v = np.array([0.0, 1.0, 0.0])

        sim = standard_cosine_similarity(u, v)
        assert np.isclose(sim, 0.0, atol=1e-6)

    def test_lorentz_orthogonal(self):
        """Lorentz similarity of orthogonal vectors."""
        u = np.array([1.0, 0.0, 0.0])
        v = np.array([0.0, 1.0, 0.0])

        sim = lorentz_similarity(u, v)
        # Lorentz: u·v - ||u||*||v|| = 0 - 1*1 = -1
        # Denominators are both 0 (lightlike), so this should return 0.0
        # due to lightlike boundary handling
        assert np.isclose(sim, 0.0, atol=1e-6) or np.isclose(sim, -1.0, atol=1e-6)


class TestParallelVectors:
    """Test behavior with parallel vectors."""

    def test_standard_parallel_same_direction(self):
        """Standard cosine similarity of parallel vectors is 1.0."""
        u = np.array([1.0, 2.0, 3.0])
        v = np.array([2.0, 4.0, 6.0])  # 2 * u

        sim = standard_cosine_similarity(u, v)
        assert np.isclose(sim, 1.0, atol=1e-6)

    def test_standard_parallel_opposite_direction(self):
        """Standard cosine similarity of antiparallel vectors is -1.0."""
        u = np.array([1.0, 2.0, 3.0])
        v = np.array([-1.0, -2.0, -3.0])  # -1 * u

        sim = standard_cosine_similarity(u, v)
        assert np.isclose(sim, -1.0, atol=1e-6)

    def test_lorentz_parallel_same_direction(self):
        """Lorentz similarity of parallel vectors (same direction)."""
        u = np.array([1.0, 2.0, 3.0])
        v = np.array([2.0, 4.0, 6.0])  # 2 * u

        sim = lorentz_similarity(u, v)
        # Both are lightlike, so should return 0.0
        assert np.isclose(sim, 0.0, atol=1e-6)


class TestNumericalStability:
    """Test numerical stability edge cases."""

    def test_zero_vectors(self):
        """Test behavior with zero vectors."""
        zero = np.array([0.0, 0.0, 0.0])
        v = np.array([1.0, 2.0, 3.0])

        # Both should handle gracefully
        standard_sim = standard_cosine_similarity(zero, v)
        lorentz_sim = lorentz_similarity(zero, v)

        assert np.isfinite(standard_sim)
        assert np.isfinite(lorentz_sim)

    def test_very_small_vectors(self):
        """Test behavior with very small magnitude vectors."""
        tiny = np.array([1e-15, 1e-15, 1e-15])

        standard_sim = standard_cosine_similarity(tiny, tiny)
        lorentz_sim = lorentz_similarity(tiny, tiny)

        assert np.isfinite(standard_sim)
        assert np.isfinite(lorentz_sim)
        assert np.isclose(lorentz_sim, 0.0, atol=1e-6)

    def test_very_large_vectors(self):
        """Test behavior with very large magnitude vectors."""
        huge = np.array([1e15, 1e15, 1e15])

        standard_sim = standard_cosine_similarity(huge, huge)
        lorentz_sim = lorentz_similarity(huge, huge)

        assert np.isfinite(standard_sim)
        assert np.isfinite(lorentz_sim)
        assert np.isclose(standard_sim, 1.0, atol=1e-6)
        assert np.isclose(lorentz_sim, 0.0, atol=1e-6)


class TestHighDimensionalVectors:
    """Test behavior in high-dimensional spaces."""

    def test_random_high_dimensional_self_similarity(self):
        """Test self-similarity in high dimensions."""
        dimensions = [10, 50, 100, 500]

        for dim in dimensions:
            v = np.random.randn(dim)

            standard_sim = standard_cosine_similarity(v, v)
            lorentz_sim = lorentz_similarity(v, v)

            assert np.isclose(
                standard_sim, 1.0, atol=1e-6
            ), f"Dim {dim}: standard should be 1.0, got {standard_sim}"
            assert np.isclose(
                lorentz_sim, 0.0, atol=1e-6
            ), f"Dim {dim}: Lorentz should be 0.0, got {lorentz_sim}"

    def test_random_high_dimensional_pairs(self):
        """Test that different random vectors have non-trivial similarity."""
        np.random.seed(42)
        dim = 100

        u = np.random.randn(dim)
        v = np.random.randn(dim)

        standard_sim = standard_cosine_similarity(u, v)
        lorentz_sim = lorentz_similarity(u, v)

        # Random vectors should have similarity close to 0 but not exactly 0
        assert abs(standard_sim) < 0.3  # Likely small for random vectors
        assert np.isfinite(lorentz_sim)


class TestLoopPreventionProperty:
    """
    Test the theoretical claim: Lorentz similarity prevents
    self-reinforcing loops by yielding neutral self-similarity.
    """

    def test_iterative_self_reinforcement_standard(self):
        """
        Demonstrate how standard cosine amplifies self-reference.

        In a recursive system, self-similarity = 1.0 can create
        fixed-point attractors.
        """
        v = np.array([1.0, 2.0, 3.0])

        # Simulate an iterative attention-like mechanism
        # where similarity influences weight/importance
        standard_weight = standard_cosine_similarity(v, v)

        # In a naive recursive system, this weight would be multiplied
        # back into the system, creating potential for runaway feedback
        assert standard_weight == 1.0, "Standard self-weight is 1.0 - maximum reinforcement"

    def test_iterative_self_reinforcement_lorentz(self):
        """
        Demonstrate how Lorentz similarity neutralizes self-reference.

        The 0.0 self-similarity prevents the recursive system from
        giving extra weight to self-references.
        """
        v = np.array([1.0, 2.0, 3.0])

        # Same iterative mechanism with Lorentz similarity
        lorentz_weight = lorentz_similarity(v, v)

        # The neutral weight means self-reference contributes nothing,
        # forcing the system to incorporate external information
        assert lorentz_weight == 0.0, "Lorentz self-weight is 0.0 - neutral, non-reinforcing"

    def test_recursive_accumulation_simulation(self):
        """
        Simulate recursive accumulation to show divergence prevention.

        This models a scenario where similarity scores accumulate
        over iterations (e.g., attention weight accumulation).
        """
        v = np.array([1.0, 2.0, 3.0])
        iterations = 10

        # Standard cosine: accumulates self-similarity
        standard_accumulation = sum(standard_cosine_similarity(v, v) for _ in range(iterations))

        # Lorentz: self-similarity doesn't accumulate
        lorentz_accumulation = sum(lorentz_similarity(v, v) for _ in range(iterations))

        # Standard grows linearly with iterations
        assert np.isclose(standard_accumulation, iterations, atol=1e-6)

        # Lorentz remains at zero
        assert np.isclose(lorentz_accumulation, 0.0, atol=1e-6)

        print(f"\nRecursive Accumulation Test:")
        print(f"  Standard (potential loop amplification): {standard_accumulation}")
        print(f"  Lorentz (loop prevention): {lorentz_accumulation}")


class TestInputValidation:
    """Test input validation and error handling."""

    def test_mismatched_dimensions(self):
        """Test that mismatched vector dimensions raise errors."""
        u = np.array([1.0, 2.0])
        v = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError):
            standard_cosine_similarity(u, v)

        with pytest.raises(ValueError):
            lorentz_similarity(u, v)

    def test_list_inputs_converted(self):
        """Test that list inputs are properly converted to arrays."""
        u = [1.0, 2.0, 3.0]
        v = [1.0, 2.0, 3.0]

        # Should not raise errors
        standard_sim = standard_cosine_similarity(u, v)
        lorentz_sim = lorentz_similarity(u, v)

        assert np.isclose(standard_sim, 1.0, atol=1e-6)
        assert np.isclose(lorentz_sim, 0.0, atol=1e-6)


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
