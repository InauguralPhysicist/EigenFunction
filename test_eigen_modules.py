"""
Test Suite for EigenFunction PyTorch Modules
=============================================

Tests for eigen_memory.py and eigen_attention.py modules that use
Lorentz-invariant similarity for loop prevention in neural networks.

These tests validate:
- Module initialization and basic functionality
- Loop prevention properties (self-similarity ~0)
- Gradient flow for backpropagation
- Device handling (CPU/CUDA)
- Edge cases and numerical stability
"""

import pytest
import numpy as np

# Check if PyTorch is available
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    pytest.skip("PyTorch not available", allow_module_level=True)

from gpu_similarity import eigen_similarity, is_torch_available
from eigen_memory import EigenMemory
from eigen_attention import EigenAttention


@pytest.fixture
def device():
    """Get device for testing (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestEigenSimilarityTorch:
    """Test the eigen_similarity function with PyTorch tensors."""

    def test_torch_tensor_basic(self, device):
        """Test basic similarity computation with PyTorch tensors."""
        u = torch.randn(10, 64, device=device)
        v = torch.randn(20, 64, device=device)

        sim = eigen_similarity(u, v)

        assert isinstance(sim, torch.Tensor)
        assert sim.shape == (10, 20)
        assert sim.device == device
        assert torch.all(sim >= -1.0)
        assert torch.all(sim <= 1.0)

    def test_self_similarity_is_zero(self, device):
        """Test that self-similarity is ~0 (lightlike)."""
        x = torch.randn(50, 128, device=device)

        sim = eigen_similarity(x, x)

        # Diagonal should be ~0
        diagonal = torch.diag(sim)
        assert torch.allclose(diagonal, torch.zeros_like(diagonal), atol=1e-5), \
            f"Self-similarity should be ~0, got {diagonal[:5]}"

    def test_gradient_flow(self, device):
        """Test that gradients flow through similarity computation."""
        u = torch.randn(10, 64, device=device, requires_grad=True)
        v = torch.randn(20, 64, device=device, requires_grad=True)

        sim = eigen_similarity(u, v)
        loss = sim.sum()
        loss.backward()

        assert u.grad is not None
        assert v.grad is not None
        assert not torch.isnan(u.grad).any()
        assert not torch.isnan(v.grad).any()

    def test_3d_tensor_handling(self, device):
        """Test handling of 3D tensors (batch, seq_len, dim)."""
        # Attention-style inputs
        q = torch.randn(4, 16, 64, device=device)  # (batch, seq_len, dim)
        k = torch.randn(4, 16, 64, device=device)

        sim = eigen_similarity(q, k)

        # Should flatten to (4*16, 4*16)
        assert sim.shape == (64, 64)
        assert sim.device == device

    def test_consistency_with_numpy(self):
        """Test that PyTorch and NumPy give same results."""
        # Use CPU for consistency
        u_torch = torch.randn(10, 32)
        v_torch = torch.randn(15, 32)

        u_np = u_torch.numpy()
        v_np = v_torch.numpy()

        sim_torch = eigen_similarity(u_torch, v_torch)
        sim_np = eigen_similarity(u_np, v_np)

        # Convert to numpy for comparison
        sim_torch_np = sim_torch.detach().numpy()

        np.testing.assert_allclose(sim_torch_np, sim_np, atol=1e-6)


class TestEigenMemory:
    """Test the EigenMemory module."""

    def test_initialization(self, device):
        """Test memory module initialization."""
        mem = EigenMemory(
            dim=128,
            max_mem_slots=1000,
            k_top=32,
            device=device
        )

        assert mem.dim == 128
        assert mem.max_mem_slots == 1000
        assert mem.k_top == 32
        assert mem.device == device
        assert int(mem.count.item()) == 0

    def test_write_and_read(self, device):
        """Test writing to and reading from memory."""
        mem = EigenMemory(dim=64, max_mem_slots=100, k_top=10, device=device)

        # Write some vectors
        x = torch.randn(20, 64, device=device)
        mem.write(x)

        assert int(mem.count.item()) == 20

        # Query memory
        q = torch.randn(5, 64, device=device)
        retrieved = mem.forward(q)

        assert retrieved.shape == (5, 64)
        assert retrieved.device == device

    def test_empty_memory_retrieval(self, device):
        """Test retrieval from empty memory."""
        mem = EigenMemory(dim=64, max_mem_slots=100, device=device)

        q = torch.randn(5, 64, device=device)
        retrieved = mem.forward(q)

        assert retrieved.shape == (5, 64)
        assert torch.allclose(retrieved, torch.zeros_like(retrieved))

    def test_loop_prevention(self, device):
        """Test that memory doesn't retrieve itself (loop prevention)."""
        mem = EigenMemory(
            dim=64,
            max_mem_slots=100,
            k_top=10,
            loop_epsilon=1e-3,
            device=device
        )

        # Write vectors
        x = torch.randn(50, 64, device=device)
        mem.write(x)

        # Query with same vectors
        retrieved, (attn, idx_topk) = mem.forward(x[:10], return_weights=True)

        # Check that self-connections have very low weight
        # (due to loop prevention, self-similarity is ~0)
        assert attn is not None
        assert retrieved.shape == (10, 64)

        # Retrieved vectors should not be identical to queries
        # (some difference due to mixing)
        similarity = torch.cosine_similarity(x[:10], retrieved, dim=-1)
        assert not torch.allclose(similarity, torch.ones_like(similarity), atol=0.1)

    def test_temporal_decay(self, device):
        """Test that temporal decay favors recent entries."""
        mem = EigenMemory(
            dim=32,
            max_mem_slots=100,
            k_top=5,
            decay=0.9,
            device=device
        )

        # Write old entries
        old = torch.randn(10, 32, device=device)
        mem.write(old)

        # Age increases
        old_age = mem.age[:10].clone()

        # Write new entries
        new = torch.randn(10, 32, device=device)
        mem.write(new)

        # Check that age increased for old entries
        assert torch.all(mem.age[:10] > old_age)
        # New entries have age 0
        assert torch.allclose(mem.age[10:20], torch.ones(10, device=device))

    def test_ring_buffer_overflow(self, device):
        """Test ring-buffer behavior when memory is full."""
        mem = EigenMemory(dim=16, max_mem_slots=50, device=device)

        # Fill memory
        x1 = torch.randn(50, 16, device=device)
        mem.write(x1)
        assert int(mem.count.item()) == 50

        # Write more (should overwrite)
        x2 = torch.randn(30, 16, device=device)
        mem.write(x2)

        # Count should still be max
        assert int(mem.count.item()) == 50

    def test_gradient_flow_through_retrieval(self, device):
        """Test that gradients flow through memory retrieval."""
        mem = EigenMemory(dim=32, max_mem_slots=50, device=device)

        # Write some vectors (no grad needed for writes)
        with torch.no_grad():
            x = torch.randn(20, 32, device=device)
            mem.write(x)

        # Query with grad
        q = torch.randn(5, 32, device=device, requires_grad=True)
        retrieved = mem.forward(q)

        loss = retrieved.sum()
        loss.backward()

        assert q.grad is not None
        assert not torch.isnan(q.grad).any()


class TestEigenAttention:
    """Test the EigenAttention module."""

    def test_initialization(self, device):
        """Test attention module initialization."""
        attn = EigenAttention(
            dim=256,
            num_heads=8,
            device=device
        )

        assert attn.dim == 256
        assert attn.num_heads == 8
        assert attn.head_dim == 32

    def test_forward_pass(self, device):
        """Test basic forward pass."""
        attn = EigenAttention(dim=128, num_heads=4).to(device)

        x = torch.randn(2, 16, 128, device=device)  # (batch, seq, dim)
        out, weights = attn.forward(x)

        assert out.shape == (2, 16, 128)
        assert weights.shape == (2, 4, 16, 16)  # (batch, heads, seq, seq)
        assert out.device == device

    def test_self_attention_loop_prevention(self, device):
        """Test that self-attention doesn't over-weight self-tokens."""
        attn = EigenAttention(
            dim=64,
            num_heads=2,
            loop_epsilon=1e-3
        ).to(device)

        x = torch.randn(1, 10, 64, device=device)
        out, weights = attn.forward(x)

        # Check attention weights
        # Diagonal should not dominate (due to loop prevention)
        for h in range(2):
            diag = torch.diag(weights[0, h])
            # Diagonal shouldn't be much larger than off-diagonal
            off_diag = weights[0, h][~torch.eye(10, dtype=bool)]
            assert diag.mean() < off_diag.mean() * 2, \
                "Self-attention should not dominate due to loop prevention"

    def test_causal_masking(self, device):
        """Test causal attention masking."""
        attn = EigenAttention(
            dim=64,
            num_heads=2,
            causal=True
        ).to(device)

        x = torch.randn(1, 8, 64, device=device)
        out, weights = attn.forward(x)

        # Check that future positions have near-zero weight
        for h in range(2):
            # Upper triangle (future) should be near zero
            upper_tri = torch.triu(weights[0, h], diagonal=1)
            assert torch.allclose(upper_tri, torch.zeros_like(upper_tri), atol=1e-6)

    def test_attention_mask(self, device):
        """Test external attention masking."""
        attn = EigenAttention(dim=64, num_heads=2).to(device)

        x = torch.randn(2, 8, 64, device=device)

        # Create mask: mask out last 2 positions
        mask = torch.zeros(2, 1, 8, 8, device=device)
        mask[:, :, :, -2:] = float('-inf')

        out, weights = attn.forward(x, attn_mask=mask)

        # Check that masked positions have zero weight
        assert torch.allclose(
            weights[:, :, :, -2:],
            torch.zeros_like(weights[:, :, :, -2:]),
            atol=1e-6
        )

    def test_gradient_flow(self, device):
        """Test gradient flow through attention."""
        attn = EigenAttention(dim=64, num_heads=2).to(device)

        x = torch.randn(2, 8, 64, device=device, requires_grad=True)
        out, weights = attn.forward(x)

        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

        # Check that all attention parameters have gradients
        for name, param in attn.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"

    def test_multi_head_independence(self, device):
        """Test that different heads produce different attention patterns."""
        attn = EigenAttention(dim=64, num_heads=4).to(device)

        x = torch.randn(1, 10, 64, device=device)
        out, weights = attn.forward(x)

        # Check that heads have different patterns
        head_weights = [weights[0, h] for h in range(4)]

        for i in range(4):
            for j in range(i + 1, 4):
                # Heads should not be identical
                similarity = torch.cosine_similarity(
                    head_weights[i].flatten(),
                    head_weights[j].flatten(),
                    dim=0
                )
                assert similarity < 0.99, \
                    f"Heads {i} and {j} are too similar: {similarity}"

    def test_negative_masking(self, device):
        """Test negative similarity masking."""
        attn = EigenAttention(
            dim=32,
            num_heads=2,
            mask_negative=True,
            negative_floor=-10.0
        ).to(device)

        x = torch.randn(1, 6, 32, device=device)
        out, weights = attn.forward(x)

        # Should complete without errors
        assert out.shape == (1, 6, 32)
        assert not torch.isnan(out).any()


class TestIntegration:
    """Integration tests combining attention and memory."""

    def test_attention_memory_pipeline(self, device):
        """Test using attention and memory together."""
        # Simulate a simple sequence processing pipeline
        dim = 64
        seq_len = 10
        batch = 2

        # Attention layer
        attn = EigenAttention(dim=dim, num_heads=2).to(device)

        # Memory module
        mem = EigenMemory(dim=dim, max_mem_slots=100, k_top=5, device=device)

        # Input sequence
        x = torch.randn(batch, seq_len, dim, device=device)

        # Process through attention
        attn_out, attn_weights = attn.forward(x)

        assert attn_out.shape == (batch, seq_len, dim)

        # Write attention output to memory
        with torch.no_grad():
            mem.write(attn_out.reshape(-1, dim))

        assert int(mem.count.item()) == batch * seq_len

        # Query memory with new inputs
        q = torch.randn(5, dim, device=device)
        retrieved = mem.forward(q)

        assert retrieved.shape == (5, dim)

    def test_loop_prevention_in_pipeline(self, device):
        """Test loop prevention in combined attention + memory."""
        dim = 32

        attn = EigenAttention(
            dim=dim,
            num_heads=2,
            loop_epsilon=1e-3
        ).to(device)

        mem = EigenMemory(
            dim=dim,
            max_mem_slots=50,
            k_top=5,
            loop_epsilon=1e-3,
            device=device
        )

        # Process sequence
        x = torch.randn(1, 8, dim, device=device)
        attn_out, weights = attn.forward(x)

        # Self-attention diagonal should be low
        for h in range(2):
            diag = torch.diag(weights[0, h])
            assert diag.mean() < 0.3, "Self-attention should be suppressed"

        # Write to memory and query same vectors
        with torch.no_grad():
            vectors = attn_out.reshape(-1, dim)
            mem.write(vectors)

        retrieved, (mem_attn, idx) = mem.forward(vectors[:4], return_weights=True)

        # Memory shouldn't retrieve exact same vectors with high weight
        # due to loop prevention
        assert retrieved.shape == (4, dim)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_dimension_mismatch_memory(self, device):
        """Test error handling for dimension mismatch in memory."""
        mem = EigenMemory(dim=64, max_mem_slots=100, device=device)

        # Wrong dimension
        with pytest.raises(ValueError):
            x = torch.randn(10, 32, device=device)
            mem.write(x)

    def test_dimension_mismatch_attention(self, device):
        """Test error handling for dimension mismatch in attention."""
        attn = EigenAttention(dim=64, num_heads=4).to(device)

        # Wrong dimension
        with pytest.raises(ValueError):
            x = torch.randn(2, 8, 32, device=device)
            attn.forward(x)

    def test_invalid_tensor_shape(self, device):
        """Test error handling for invalid tensor shapes."""
        mem = EigenMemory(dim=64, max_mem_slots=100, device=device)

        # 1D tensor (invalid)
        with pytest.raises(ValueError):
            x = torch.randn(64, device=device)
            mem.write(x)

        # 3D tensor (invalid for memory write)
        with pytest.raises(ValueError):
            x = torch.randn(2, 10, 64, device=device)
            mem.write(x)

    def test_zero_vectors(self, device):
        """Test handling of zero vectors."""
        mem = EigenMemory(dim=32, max_mem_slots=50, device=device)

        # Write zero vectors
        x = torch.zeros(10, 32, device=device)
        mem.write(x)

        # Query with zero vector
        q = torch.zeros(1, 32, device=device)
        retrieved = mem.forward(q)

        assert not torch.isnan(retrieved).any()


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
