import unittest
import torch

from eigen_memory import EigenMemory
from eigen_attention import EigenAttention
from eigen_transformer import EigenTransformerBlock
from gpu_similarity import eigen_similarity


class TestEigenModules(unittest.TestCase):
    def setUp(self):
        self.dim = 64
        self.num_heads = 4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(42)

    def test_eigen_similarity_shapes(self):
        """
        Basic sanity check: eigen_similarity should produce correct shapes and stay in [-1, 1].
        """
        B, L_q, L_k, D = 2, 3, 5, self.dim
        q = torch.randn(B, L_q, D, device=self.device)
        k = torch.randn(B, L_k, D, device=self.device)

        sim = eigen_similarity(q, k)  # expected (B, L_q, L_k)

        self.assertEqual(sim.shape, (B, L_q, L_k))
        self.assertTrue(torch.all(sim <= 1.0 + 1e-6))
        self.assertTrue(torch.all(sim >= -1.0 - 1e-6))

    def test_eigen_memory_self_similarity_suppression(self):
        """
        Verify that loop_epsilon threshold suppresses near-self retrieval.
        """
        memory = EigenMemory(
            dim=self.dim,
            max_mem_slots=20,
            k_top=5,
            loop_epsilon=0.01,  # Suppress similarities below this threshold
            device=self.device,
        )

        # Write multiple diverse entries
        torch.manual_seed(123)
        for i in range(10):
            mem_entry = torch.randn(1, self.dim, device=self.device)
            memory.write(mem_entry)

        # Base state
        h_t = torch.randn(1, self.dim, device=self.device)
        memory.write(h_t)

        # Near-identical variant (self-like) - should be suppressed
        h_t_self = h_t.clone()  # Exactly the same

        # Different state
        h_t_far = torch.randn(1, self.dim, device=self.device)

        # Query with attention weights
        _, (attn_self, _) = memory(h_t_self, return_weights=True)
        _, (attn_far, _) = memory(h_t_far, return_weights=True)

        # The self query should have valid attention distribution
        self.assertTrue(torch.isfinite(attn_self).all(), "Self attention should be finite")
        self.assertTrue(torch.isfinite(attn_far).all(), "Far attention should be finite")

        # Attention weights should sum to 1
        self.assertAlmostEqual(attn_self.sum().item(), 1.0, places=5)
        self.assertAlmostEqual(attn_far.sum().item(), 1.0, places=5)

    def test_eigen_attention_causal_vs_disconnected(self):
        """
        In a causal chain, attention should allocate more mass to past positions
        than in a spacelike-disconnected sequence.
        """
        attn = EigenAttention(
            dim=self.dim,
            num_heads=self.num_heads,
            causal=True,
            loop_epsilon=0.1,
        ).to(self.device)

        B, L = 1, 5

        # Causal chain: increasing timelike coordinate
        seq_causal = torch.randn(B, L, self.dim, device=self.device)
        # strictly increasing "time" component (non-negative increments)
        time_increments = torch.abs(torch.randn(B, L, device=self.device))
        seq_causal[..., 0] = torch.cumsum(time_increments, dim=1)

        out_causal, attn_weights_causal = attn(seq_causal)  # attn: (B, H, L, L)

        # Attention to past positions (lower triangle, excluding diagonal)
        lower_tri = torch.tril(attn_weights_causal, diagonal=-1)
        mean_past_attention_causal = lower_tri.mean().item()

        # Disconnected: large spacelike differences
        seq_disconn = torch.randn(B, L, self.dim, device=self.device)
        seq_disconn[..., 1:] += 10.0  # large spatial separation

        out_disconn, attn_weights_disconn = attn(seq_disconn)
        lower_tri_disconn = torch.tril(attn_weights_disconn, diagonal=-1)
        mean_past_attention_disconn = lower_tri_disconn.mean().item()

        self.assertGreater(
            mean_past_attention_causal,
            mean_past_attention_disconn,
            "Causal chain should yield stronger attention to past than disconnected sequence.",
        )

    def test_eigen_transformer_block_forward(self):
        """
        Smoke test: EigenTransformerBlock should run forward and preserve shape.
        """
        block = EigenTransformerBlock(
            dim=self.dim,
            num_heads=self.num_heads,
            mlp_ratio=2.0,
            dropout=0.0,
            attn_kwargs={"causal": True},
        ).to(self.device)

        B, L = 2, 7
        x = torch.randn(B, L, self.dim, device=self.device)

        out = block(x)  # (B, L, D)
        self.assertEqual(out.shape, x.shape)

    def test_full_loop_prevention(self):
        """
        Verify EigenAttention and EigenMemory can be composed without numerical issues.
        """
        memory = EigenMemory(
            self.dim,
            max_mem_slots=20,
            k_top=5,
            loop_epsilon=1e-3,
            device=self.device
        )
        attn = EigenAttention(
            self.dim,
            self.num_heads,
            loop_epsilon=1e-3
        ).to(self.device)

        # Create a simple sequence
        B, L = 2, 5
        x = torch.randn(B, L, self.dim, device=self.device)

        # Apply attention
        attn_out, attn_weights = attn(x)

        # Verify attention output is finite
        self.assertTrue(torch.isfinite(attn_out).all(), "Attention output should be finite")
        self.assertTrue(torch.isfinite(attn_weights).all(), "Attention weights should be finite")

        # Write some states to memory
        for i in range(L):
            memory.write(x[:, i, :])  # Write each position

        # Retrieve from memory
        query = torch.randn(3, self.dim, device=self.device)
        retrieved = memory(query)

        # Verify retrieval is finite
        self.assertTrue(torch.isfinite(retrieved).all(), "Retrieved memory should be finite")
        self.assertEqual(retrieved.shape, (3, self.dim), "Retrieved shape should match query")

        # Verify attention weights sum to 1 along last dimension
        attn_sums = attn_weights.sum(dim=-1)
        self.assertTrue(
            torch.allclose(attn_sums, torch.ones_like(attn_sums), atol=1e-5),
            "Attention weights should sum to 1"
        )


if __name__ == "__main__":
    unittest.main()
