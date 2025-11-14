"""
Test Suite for GPU-Accelerated Lorentz-Invariant Similarity
============================================================

This test suite validates GPU implementations and ensures consistency
with CPU implementations.

Tests can run with or without GPU:
- If GPU is available, tests validate GPU functionality
- If GPU is not available, tests validate graceful fallback to CPU
"""

import numpy as np
import pytest
import gpu_similarity as gpu_sim
from similarity import lorentz_similarity, standard_cosine_similarity


class TestGPUAvailability:
    """Test GPU availability detection and setup."""

    def test_gpu_check_returns_bool(self):
        """GPU availability check should return boolean."""
        available = gpu_sim.is_gpu_available()
        assert isinstance(available, bool)

    def test_array_module_detection(self):
        """Test array module detection for numpy arrays."""
        arr = np.array([1.0, 2.0, 3.0])
        module = gpu_sim.get_array_module(arr)
        assert module == np

    def test_to_cpu_with_numpy(self):
        """Test CPU transfer with numpy array."""
        arr = np.array([1.0, 2.0, 3.0])
        cpu_arr = gpu_sim.to_cpu(arr)
        assert isinstance(cpu_arr, np.ndarray)
        np.testing.assert_array_equal(arr, cpu_arr)


class TestGPUCPUConsistency:
    """
    Test that GPU implementations give identical results to CPU.

    These tests are critical for validating correctness.
    """

    def test_lorentz_similarity_self_consistency(self):
        """GPU and CPU Lorentz similarity should match for self-similarity."""
        vectors = [
            np.array([1.0, 0.0, 0.0]),
            np.array([3.0, 4.0]),
            np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            np.random.randn(10),
            np.random.randn(100),
        ]

        for v in vectors:
            cpu_sim = lorentz_similarity(v, v)
            gpu_sim_result = gpu_sim.lorentz_similarity_gpu(v, v)

            assert np.isclose(
                cpu_sim, gpu_sim_result, atol=1e-9
            ), f"CPU: {cpu_sim}, GPU: {gpu_sim_result}"

            # Both should be ~0.0 for self-similarity
            assert np.isclose(gpu_sim_result, 0.0, atol=1e-6)

    def test_lorentz_similarity_pair_consistency(self):
        """GPU and CPU Lorentz similarity should match for vector pairs."""
        test_pairs = [
            (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),
            (np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])),
            (np.random.randn(50), np.random.randn(50)),
            (np.random.randn(200), np.random.randn(200)),
        ]

        for u, v in test_pairs:
            cpu_sim = lorentz_similarity(u, v)
            gpu_sim_result = gpu_sim.lorentz_similarity_gpu(u, v)

            assert np.isclose(
                cpu_sim, gpu_sim_result, atol=1e-9
            ), f"Shape {u.shape}: CPU: {cpu_sim}, GPU: {gpu_sim_result}"

    def test_standard_cosine_consistency(self):
        """GPU and CPU standard cosine similarity should match."""
        test_pairs = [
            (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),
            (np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])),
            (np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0])),
        ]

        for u, v in test_pairs:
            cpu_sim = standard_cosine_similarity(u, v)
            gpu_sim_result = gpu_sim.standard_cosine_similarity_gpu(u, v)

            assert np.isclose(
                cpu_sim, gpu_sim_result, atol=1e-9
            ), f"CPU: {cpu_sim}, GPU: {gpu_sim_result}"


class TestBatchOperations:
    """Test GPU batch processing capabilities."""

    def test_batch_self_similarity(self):
        """Batch processing should return 0.0 for identical vectors."""
        N = 100
        D = 64

        # Create batch where each pair is identical
        U = np.random.randn(N, D)
        V = U.copy()

        similarities = gpu_sim.lorentz_similarity_batch_gpu(U, V)

        assert similarities.shape == (N,)
        assert np.allclose(
            similarities, 0.0, atol=1e-6
        ), f"All self-similarities should be ~0.0, got {similarities[:5]}"

    def test_batch_consistency_with_loop(self):
        """Batch processing should match individual computations."""
        N = 50
        D = 32

        U = np.random.randn(N, D)
        V = np.random.randn(N, D)

        # Batch computation
        batch_sims = gpu_sim.lorentz_similarity_batch_gpu(U, V)

        # Individual computations
        individual_sims = np.array([lorentz_similarity(U[i], V[i]) for i in range(N)])

        np.testing.assert_allclose(batch_sims, individual_sims, atol=1e-9)

    def test_batch_different_sizes(self):
        """Test batch processing with various sizes."""
        dimensions = [8, 32, 128, 512]
        batch_sizes = [1, 10, 100]

        for D in dimensions:
            for N in batch_sizes:
                U = np.random.randn(N, D)
                V = np.random.randn(N, D)

                similarities = gpu_sim.lorentz_similarity_batch_gpu(U, V)

                assert similarities.shape == (N,)
                assert np.all(np.isfinite(similarities))
                assert np.all(similarities >= -1.0)
                assert np.all(similarities <= 1.0)

    def test_batch_input_validation(self):
        """Test that batch processing validates inputs."""
        # Mismatched shapes
        U = np.random.randn(10, 32)
        V = np.random.randn(5, 32)

        with pytest.raises(ValueError):
            gpu_sim.lorentz_similarity_batch_gpu(U, V)

        # Wrong dimensionality
        U_1d = np.random.randn(10)
        with pytest.raises(ValueError):
            gpu_sim.lorentz_similarity_batch_gpu(U_1d, U_1d)


class TestMatrixOperations:
    """Test pairwise similarity matrix computations."""

    def test_matrix_self_similarity_diagonal(self):
        """Self-similarity matrix should have ~0.0 on diagonal."""
        N = 20
        D = 32

        U = np.random.randn(N, D)

        # Compute self-similarity matrix
        sim_matrix = gpu_sim.lorentz_similarity_matrix_gpu(U, U)

        assert sim_matrix.shape == (N, N)

        # Diagonal should be ~0.0 (self-similarity)
        diagonal = np.diag(sim_matrix)
        assert np.allclose(diagonal, 0.0, atol=1e-6), f"Diagonal should be ~0.0, got {diagonal[:5]}"

    def test_matrix_symmetry(self):
        """Self-similarity matrix should be symmetric."""
        N = 15
        D = 16

        U = np.random.randn(N, D)

        sim_matrix = gpu_sim.lorentz_similarity_matrix_gpu(U, U)

        # Check symmetry
        np.testing.assert_allclose(sim_matrix, sim_matrix.T, atol=1e-9)

    def test_matrix_rectangular(self):
        """Test similarity matrix for different sized sets."""
        N = 10
        M = 15
        D = 24

        U = np.random.randn(N, D)
        V = np.random.randn(M, D)

        sim_matrix = gpu_sim.lorentz_similarity_matrix_gpu(U, V)

        assert sim_matrix.shape == (N, M)
        assert np.all(np.isfinite(sim_matrix))
        assert np.all(sim_matrix >= -1.0)
        assert np.all(sim_matrix <= 1.0)

    def test_matrix_consistency_with_pairwise(self):
        """Matrix computation should match pairwise computations."""
        N = 10
        M = 8
        D = 16

        U = np.random.randn(N, D)
        V = np.random.randn(M, D)

        # Matrix computation
        sim_matrix = gpu_sim.lorentz_similarity_matrix_gpu(U, V)

        # Pairwise computations
        for i in range(N):
            for j in range(M):
                expected = lorentz_similarity(U[i], V[j])
                actual = sim_matrix[i, j]
                assert np.isclose(
                    expected, actual, atol=1e-9
                ), f"Mismatch at [{i},{j}]: {expected} vs {actual}"

    def test_matrix_input_validation(self):
        """Test matrix computation input validation."""
        # Dimension mismatch
        U = np.random.randn(10, 32)
        V = np.random.randn(10, 16)

        with pytest.raises(ValueError):
            gpu_sim.lorentz_similarity_matrix_gpu(U, V)

        # Wrong dimensionality
        U_1d = np.random.randn(10)
        with pytest.raises(ValueError):
            gpu_sim.lorentz_similarity_matrix_gpu(U_1d)


class TestAttentionMechanismSimulation:
    """
    Test GPU module in attention-like scenarios.

    This validates the practical use case of loop prevention
    in transformer-style attention mechanisms.
    """

    def test_self_attention_no_self_loops(self):
        """
        Self-attention with Lorentz similarity should not over-weight self.

        In a standard attention mechanism, a token attending to itself
        gets maximum weight (1.0), potentially creating loops.
        With Lorentz similarity, self-attention weight is 0.0.
        """
        # Simulate token embeddings
        num_tokens = 50
        embedding_dim = 128

        embeddings = np.random.randn(num_tokens, embedding_dim)

        # Compute attention scores (queries = keys for self-attention)
        attention_scores = gpu_sim.lorentz_similarity_matrix_gpu(embeddings, embeddings)

        # Diagonal (self-attention) should be ~0.0
        diagonal = np.diag(attention_scores)
        assert np.allclose(diagonal, 0.0, atol=1e-6), "Self-attention should be neutralized"

        # Off-diagonal should have meaningful values
        off_diagonal = attention_scores[np.triu_indices(num_tokens, k=1)]
        assert not np.allclose(off_diagonal, 0.0), "Cross-attention should have meaningful scores"

    def test_query_key_attention(self):
        """Test query-key attention pattern."""
        num_queries = 32
        num_keys = 64
        embedding_dim = 96

        queries = np.random.randn(num_queries, embedding_dim)
        keys = np.random.randn(num_keys, embedding_dim)

        attention_scores = gpu_sim.lorentz_similarity_matrix_gpu(queries, keys)

        assert attention_scores.shape == (num_queries, num_keys)
        assert np.all(np.isfinite(attention_scores))


class TestNumericalStability:
    """Test GPU numerical stability with edge cases."""

    def test_zero_vectors_gpu(self):
        """Test GPU handling of zero vectors."""
        zero = np.array([0.0, 0.0, 0.0])
        v = np.array([1.0, 2.0, 3.0])

        gpu_sim_result = gpu_sim.lorentz_similarity_gpu(zero, v)
        assert np.isfinite(gpu_sim_result)

    def test_tiny_vectors_gpu(self):
        """Test GPU handling of very small vectors."""
        tiny = np.array([1e-15, 1e-15, 1e-15])

        gpu_sim_result = gpu_sim.lorentz_similarity_gpu(tiny, tiny)
        assert np.isfinite(gpu_sim_result)
        assert np.isclose(gpu_sim_result, 0.0, atol=1e-6)

    def test_huge_vectors_gpu(self):
        """Test GPU handling of very large vectors."""
        huge = np.array([1e15, 1e15, 1e15])

        gpu_sim_result = gpu_sim.lorentz_similarity_gpu(huge, huge)
        assert np.isfinite(gpu_sim_result)
        assert np.isclose(gpu_sim_result, 0.0, atol=1e-6)

    def test_batch_with_zeros(self):
        """Test batch processing with some zero vectors."""
        N = 10
        D = 16

        U = np.random.randn(N, D)
        V = np.random.randn(N, D)

        # Set some vectors to zero
        U[0] = 0.0
        V[5] = 0.0

        similarities = gpu_sim.lorentz_similarity_batch_gpu(U, V)
        assert np.all(np.isfinite(similarities))


class TestAutoSelection:
    """Test automatic GPU/CPU selection."""

    def test_auto_selection_runs(self):
        """Test that auto selection function runs without errors."""
        u = np.array([1.0, 2.0, 3.0])
        v = np.array([4.0, 5.0, 6.0])

        # Should run whether or not GPU is available
        sim = gpu_sim.lorentz_similarity_auto(u, v)
        assert np.isfinite(sim)

    def test_auto_selection_consistency(self):
        """Test that auto selection gives correct results."""
        u = np.array([1.0, 2.0, 3.0])

        # Self-similarity should be 0.0
        sim = gpu_sim.lorentz_similarity_auto(u, u)
        assert np.isclose(sim, 0.0, atol=1e-6)


class TestPerformanceCharacteristics:
    """
    Test performance-related characteristics.

    These don't test speed directly but validate that batch operations
    can handle large inputs.
    """

    def test_large_batch_processing(self):
        """Test that large batches can be processed."""
        N = 1000
        D = 256

        U = np.random.randn(N, D)
        V = np.random.randn(N, D)

        # Should complete without errors
        similarities = gpu_sim.lorentz_similarity_batch_gpu(U, V)

        assert similarities.shape == (N,)
        assert np.all(np.isfinite(similarities))

    def test_large_matrix_computation(self):
        """Test that large similarity matrices can be computed."""
        N = 200
        M = 300
        D = 128

        U = np.random.randn(N, D)
        V = np.random.randn(M, D)

        # Should complete without errors
        sim_matrix = gpu_sim.lorentz_similarity_matrix_gpu(U, V)

        assert sim_matrix.shape == (N, M)
        assert np.all(np.isfinite(sim_matrix))

    @pytest.mark.slow
    def test_very_large_dimensions(self):
        """Test with very high dimensional vectors (marked as slow)."""
        N = 100
        D = 4096  # Very high dimensional (e.g., large transformer embeddings)

        U = np.random.randn(N, D)

        # Self-similarity matrix
        sim_matrix = gpu_sim.lorentz_similarity_matrix_gpu(U, U)

        assert sim_matrix.shape == (N, N)
        diagonal = np.diag(sim_matrix)
        assert np.allclose(diagonal, 0.0, atol=1e-6)


if __name__ == "__main__":
    # Run tests with verbose output
    # Use -v for verbose, -s to show print statements
    # Use -m "not slow" to skip slow tests
    pytest.main([__file__, "-v", "-s", "-m", "not slow"])
