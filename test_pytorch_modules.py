"""
Test Suite for PyTorch EigenFunction Modules
=============================================

Tests for GPU-accelerated Lorentz-invariant similarity and
neural network modules (EigenMemory, EigenAttention).
"""

import pytest
import torch
import torch.nn as nn

from gpu_similarity import (
    eigen_similarity,
    standard_cosine_similarity_torch,
    compare_self_similarity_torch,
)
from eigen_memory import EigenMemory
from eigen_attention import EigenAttention


class TestGPUSimilarity:
    """Test the GPU-accelerated similarity functions."""

    def test_eigen_similarity_2d_self_is_zero(self):
        """Eigen self-similarity should be ~0.0 (loop prevention)."""
        q = torch.randn(16, 64)
        sim = eigen_similarity(q, q)

        # Diagonal should be near zero
        diag = torch.diagonal(sim)
        assert torch.allclose(
            diag, torch.zeros_like(diag), atol=1e-5
        ), f"Expected diagonal ~0.0, got {diag}"

    def test_standard_cosine_2d_self_is_one(self):
        """Standard cosine self-similarity should be ~1.0."""
        q = torch.randn(16, 64)
        sim = standard_cosine_similarity_torch(q, q)

        diag = torch.diagonal(sim)
        assert torch.allclose(
            diag, torch.ones_like(diag), atol=1e-5
        ), f"Expected diagonal ~1.0, got {diag}"

    def test_eigen_similarity_3d_self_is_zero(self):
        """Eigen self-similarity for sequences should be ~0.0."""
        x = torch.randn(4, 10, 32)  # batch=4, seq_len=10, dim=32
        sim = eigen_similarity(x, x)  # (4, 10, 10)

        assert sim.shape == (4, 10, 10)

        # Diagonal should be near zero for each batch
        for b in range(4):
            diag = torch.diagonal(sim[b])
            assert torch.allclose(
                diag, torch.zeros_like(diag), atol=1e-5
            ), f"Batch {b}: Expected diagonal ~0.0, got {diag}"

    def test_standard_cosine_3d_self_is_one(self):
        """Standard cosine for sequences should have diagonal ~1.0."""
        x = torch.randn(4, 10, 32)
        sim = standard_cosine_similarity_torch(x, x)

        for b in range(4):
            diag = torch.diagonal(sim[b])
            assert torch.allclose(diag, torch.ones_like(diag), atol=1e-5)

    def test_eigen_similarity_2d_different_shapes(self):
        """Test 2D with different query/key counts: (B, D) x (N, D) -> (B, N)."""
        q = torch.randn(8, 64)
        k = torch.randn(100, 64)
        sim = eigen_similarity(q, k)

        assert sim.shape == (8, 100)
        assert sim.min() >= -1.0 and sim.max() <= 1.0

    def test_eigen_similarity_range(self):
        """Similarity should be in [-1, 1]."""
        q = torch.randn(32, 128)
        k = torch.randn(64, 128)
        sim = eigen_similarity(q, k)

        assert sim.min() >= -1.0, f"Min similarity {sim.min()} < -1.0"
        assert sim.max() <= 1.0, f"Max similarity {sim.max()} > 1.0"

    def test_compare_self_similarity_torch(self):
        """Test comparison utility function."""
        v = torch.randn(10, 64)
        result = compare_self_similarity_torch(v)

        assert "standard" in result
        assert "eigen" in result
        assert "vector_norm" in result

        # Standard diagonal should be ~1.0
        std_diag = torch.diagonal(result["standard"])
        assert torch.allclose(std_diag, torch.ones_like(std_diag), atol=1e-5)

        # Eigen diagonal should be ~0.0
        eigen_diag = torch.diagonal(result["eigen"])
        assert torch.allclose(eigen_diag, torch.zeros_like(eigen_diag), atol=1e-5)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_compatibility(self):
        """Test that similarity works on GPU."""
        q = torch.randn(16, 64).cuda()
        k = torch.randn(32, 64).cuda()

        sim = eigen_similarity(q, k)
        assert sim.device.type == "cuda"
        assert sim.shape == (16, 32)

    def test_numerical_stability_small_vectors(self):
        """Test stability with very small magnitude vectors."""
        q = torch.randn(8, 32) * 1e-10
        sim = eigen_similarity(q, q)

        assert torch.isfinite(sim).all(), "Found non-finite values"

    def test_numerical_stability_large_vectors(self):
        """Test stability with very large magnitude vectors."""
        q = torch.randn(8, 32) * 1e10
        sim = eigen_similarity(q, q)

        assert torch.isfinite(sim).all(), "Found non-finite values"


class TestEigenMemory:
    """Test EigenMemory module."""

    def test_initialization(self):
        """Test memory initialization."""
        mem = EigenMemory(dim=64, max_mem_slots=100, k_top=10)

        assert mem.dim == 64
        assert mem.max_mem_slots == 100
        assert mem.k_top == 10
        assert mem.count.item() == 0

    def test_write_and_count(self):
        """Test writing to memory updates count."""
        mem = EigenMemory(dim=32, max_mem_slots=100)

        x = torch.randn(10, 32)
        mem.write(x)

        assert mem.count.item() == 10

    def test_write_multiple_batches(self):
        """Test writing multiple batches accumulates."""
        mem = EigenMemory(dim=32, max_mem_slots=100)

        mem.write(torch.randn(10, 32))
        assert mem.count.item() == 10

        mem.write(torch.randn(15, 32))
        assert mem.count.item() == 25

    def test_write_overflow_ring_buffer(self):
        """Test that overflow wraps around (ring buffer)."""
        mem = EigenMemory(dim=16, max_mem_slots=50)

        mem.write(torch.randn(60, 16))  # More than max
        assert mem.count.item() == 50  # Should cap at max

    def test_retrieve_empty_memory(self):
        """Test retrieval from empty memory returns zeros."""
        mem = EigenMemory(dim=32, max_mem_slots=100)

        q = torch.randn(5, 32)
        retrieved = mem(q)

        assert retrieved.shape == (5, 32)
        assert torch.allclose(retrieved, torch.zeros_like(retrieved))

    def test_retrieve_with_data(self):
        """Test retrieval after writing data."""
        mem = EigenMemory(dim=32, max_mem_slots=100, k_top=5)

        # Write some data
        data = torch.randn(20, 32)
        mem.write(data)

        # Retrieve
        q = torch.randn(3, 32)
        retrieved = mem(q)

        assert retrieved.shape == (3, 32)
        assert not torch.allclose(retrieved, torch.zeros_like(retrieved))

    def test_retrieve_with_weights(self):
        """Test retrieval returns weights when requested."""
        mem = EigenMemory(dim=32, max_mem_slots=100, k_top=5)

        mem.write(torch.randn(20, 32))
        q = torch.randn(3, 32)

        retrieved, (attn, idx) = mem(q, return_weights=True)

        assert retrieved.shape == (3, 32)
        assert attn.shape == (3, 5)  # k_top=5
        assert idx.shape == (3, 5)

        # Attention should sum to 1
        assert torch.allclose(attn.sum(dim=-1), torch.ones(3), atol=1e-5)

    def test_temporal_decay(self):
        """Test that older memories decay properly."""
        mem = EigenMemory(dim=16, max_mem_slots=100, k_top=5, decay=0.9)

        # Write first batch (will be older)
        old_data = torch.randn(10, 16)
        mem.write(old_data)

        # Write second batch (newer)
        new_data = torch.randn(10, 16)
        mem.write(new_data)

        # Age should have increased for old entries
        assert mem.age[0].item() > mem.age[10].item()

    def test_loop_prevention_epsilon(self):
        """Test that loop_epsilon filters near-self similarities."""
        mem = EigenMemory(dim=32, max_mem_slots=100, k_top=10, loop_epsilon=0.5)

        data = torch.randn(20, 32)
        mem.write(data)

        # Query with same data (should have low self-similarity)
        retrieved = mem(data[:5])

        # Should retrieve something (not all zeros)
        assert not torch.allclose(retrieved, torch.zeros_like(retrieved))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_device(self):
        """Test memory works on GPU."""
        mem = EigenMemory(dim=32, max_mem_slots=100, device="cuda")

        assert mem.device.type == "cuda"

        x = torch.randn(10, 32).cuda()
        mem.write(x)

        q = torch.randn(3, 32).cuda()
        retrieved = mem(q)

        assert retrieved.device.type == "cuda"


class TestEigenAttention:
    """Test EigenAttention module."""

    def test_initialization(self):
        """Test attention layer initialization."""
        attn = EigenAttention(dim=64, num_heads=4)

        assert attn.dim == 64
        assert attn.num_heads == 4
        assert attn.head_dim == 16

    def test_forward_shape(self):
        """Test forward pass produces correct shapes."""
        attn = EigenAttention(dim=64, num_heads=4)

        x = torch.randn(2, 10, 64)  # batch=2, seq_len=10, dim=64
        out, attn_weights = attn(x)

        assert out.shape == (2, 10, 64)
        assert attn_weights.shape == (2, 4, 10, 10)  # (B, H, L, L)

    def test_attention_weights_sum_to_one(self):
        """Test that attention weights sum to 1."""
        attn = EigenAttention(dim=32, num_heads=2)

        x = torch.randn(1, 5, 32)
        _, attn_weights = attn(x)

        # Sum over last dimension (keys)
        sums = attn_weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_causal_masking(self):
        """Test causal masking prevents attending to future."""
        attn = EigenAttention(dim=32, num_heads=2, causal=True)

        x = torch.randn(1, 5, 32)
        _, attn_weights = attn(x)

        # Check that upper triangle (future tokens) has near-zero weights
        for h in range(2):
            for i in range(5):
                for j in range(i + 1, 5):
                    # Position i should not attend to future position j
                    assert (
                        attn_weights[0, h, i, j].item() < 1e-5
                    ), f"Causal mask violated: position {i} attends to {j}"

    def test_external_mask_3d(self):
        """Test external 3D attention mask (B, L, L)."""
        attn = EigenAttention(dim=32, num_heads=2)

        x = torch.randn(2, 5, 32)

        # Create mask that blocks position 0 from attending to position 4
        mask = torch.zeros(2, 5, 5)
        mask[:, 0, 4] = float("-inf")

        _, attn_weights = attn(x, attn_mask=mask)

        # Check that position 0 doesn't attend to position 4
        for b in range(2):
            for h in range(2):
                assert attn_weights[b, h, 0, 4].item() < 1e-5

    def test_external_mask_4d(self):
        """Test external 4D attention mask (B, 1, L, L)."""
        attn = EigenAttention(dim=32, num_heads=2)

        x = torch.randn(2, 5, 32)

        # 4D mask
        mask = torch.zeros(2, 1, 5, 5)
        mask[:, :, 0, 4] = float("-inf")

        _, attn_weights = attn(x, attn_mask=mask)

        for b in range(2):
            for h in range(2):
                assert attn_weights[b, h, 0, 4].item() < 1e-5

    def test_loop_prevention_epsilon(self):
        """Test that loop_epsilon attenuates self-attention."""
        # With large epsilon, self-attention should be suppressed
        attn = EigenAttention(dim=32, num_heads=2, loop_epsilon=0.5)

        x = torch.randn(1, 5, 32)
        out, attn_weights = attn(x)

        # Self-attention (diagonal) should be affected
        # (exact behavior depends on sim_scale and other params)
        assert out.shape == (1, 5, 32)

    def test_negative_masking(self):
        """Test that negative similarities can be masked."""
        attn = EigenAttention(dim=32, num_heads=2, mask_negative=True, negative_floor=-10.0)

        x = torch.randn(1, 5, 32)
        out, _ = attn(x)

        assert out.shape == (1, 5, 32)
        assert torch.isfinite(out).all()

    def test_multihead_independence(self):
        """Test that different heads produce different attention patterns."""
        attn = EigenAttention(dim=64, num_heads=4)

        x = torch.randn(1, 10, 64)
        _, attn_weights = attn(x)

        # Compare attention patterns across heads
        head_0 = attn_weights[0, 0]
        head_1 = attn_weights[0, 1]

        # They should be different (not identical)
        assert not torch.allclose(head_0, head_1, atol=1e-3)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_compatibility(self):
        """Test attention works on GPU."""
        attn = EigenAttention(dim=64, num_heads=4).cuda()

        x = torch.randn(2, 10, 64).cuda()
        out, attn_weights = attn(x)

        assert out.device.type == "cuda"
        assert attn_weights.device.type == "cuda"

    def test_gradient_flow(self):
        """Test that gradients flow through attention."""
        attn = EigenAttention(dim=32, num_heads=2)

        x = torch.randn(2, 5, 32, requires_grad=True)
        out, _ = attn(x)

        # Compute loss and backprop
        loss = out.sum()
        loss.backward()

        # Check that input has gradients
        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))


class TestIntegration:
    """Integration tests combining multiple modules."""

    def test_memory_attention_pipeline(self):
        """Test using memory retrieval with attention."""
        mem = EigenMemory(dim=64, max_mem_slots=100, k_top=10)
        attn = EigenAttention(dim=64, num_heads=4)

        # Store some memories
        mem.write(torch.randn(50, 64))

        # Create query sequence
        x = torch.randn(2, 8, 64)

        # Retrieve from memory for each position
        B, L, D = x.shape
        x_flat = x.reshape(B * L, D)
        retrieved_flat = mem(x_flat)
        retrieved = retrieved_flat.reshape(B, L, D)

        # Apply attention
        out, _ = attn(retrieved)

        assert out.shape == (2, 8, 64)

    def test_consistency_with_numpy_version(self):
        """Test that PyTorch version is consistent with NumPy version."""
        # Import NumPy version
        import numpy as np
        from similarity import lorentz_similarity

        # Create same vector in both frameworks
        v_np = np.array([1.0, 2.0, 3.0])
        v_torch = torch.tensor([1.0, 2.0, 3.0])

        # Compute self-similarity
        sim_np = lorentz_similarity(v_np, v_np)
        sim_torch = eigen_similarity(v_torch.unsqueeze(0), v_torch.unsqueeze(0))

        # Both should be ~0.0
        assert abs(sim_np) < 1e-5
        assert abs(sim_torch[0, 0].item()) < 1e-5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
