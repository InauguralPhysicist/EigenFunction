"""
GPU-Accelerated Lorentz Similarity Examples
============================================

This script demonstrates the GPU acceleration capabilities and performance
benefits for large-scale similarity computations.

The examples show:
1. Basic GPU usage
2. Batch processing for efficiency
3. Attention mechanism simulation
4. Performance comparison (when GPU is available)
5. Automatic fallback to CPU

Usage:
------
    python examples_gpu.py
"""

import numpy as np
import time
from typing import Tuple

# Import both CPU and GPU implementations
from similarity import lorentz_similarity, standard_cosine_similarity
import gpu_similarity as gpu_sim


def print_section(title: str):
    """Print formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def check_gpu_status():
    """Check and display GPU availability."""
    print_section("GPU Status Check")

    if gpu_sim.is_gpu_available():
        print("✓ GPU (CUDA) is available!")
        try:
            import cupy as cp
            device = cp.cuda.Device()
            print(f"  Device: {device}")
            print(f"  Compute Capability: {device.compute_capability}")
            mem_info = cp.cuda.Device().mem_info
            print(f"  Memory: {mem_info[1] / 1e9:.2f} GB total")
        except Exception as e:
            print(f"  (Could not get device details: {e})")
    else:
        print("⚠ GPU not available - using CPU fallback")
        print("  To enable GPU: pip install cupy-cuda11x (or cupy-cuda12x)")


def example_basic_usage():
    """Demonstrate basic GPU-accelerated similarity computation."""
    print_section("Example 1: Basic GPU Usage")

    u = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    v = np.array([2.0, 3.0, 4.0, 5.0, 6.0])

    print(f"Vector u: {u}")
    print(f"Vector v: {v}")
    print()

    # Compute using GPU
    sim_lorentz = gpu_sim.lorentz_similarity_gpu(u, v)
    sim_standard = gpu_sim.standard_cosine_similarity_gpu(u, v)

    print(f"Lorentz similarity (GPU):   {sim_lorentz:.6f}")
    print(f"Standard cosine sim (GPU):  {sim_standard:.6f}")
    print()

    # Self-similarity demonstration
    print("Self-similarity (u with itself):")
    self_sim_lorentz = gpu_sim.lorentz_similarity_gpu(u, u)
    self_sim_standard = gpu_sim.standard_cosine_similarity_gpu(u, u)

    print(f"  Lorentz:  {self_sim_lorentz:.6f}  (prevents loops!)")
    print(f"  Standard: {self_sim_standard:.6f}  (can amplify loops)")


def example_batch_processing():
    """Demonstrate efficient batch processing on GPU."""
    print_section("Example 2: Batch Processing")

    batch_size = 1000
    dimension = 128

    print(f"Processing {batch_size} vector pairs of dimension {dimension}...")
    print()

    # Generate random vectors
    np.random.seed(42)
    U = np.random.randn(batch_size, dimension)
    V = np.random.randn(batch_size, dimension)

    # Batch processing on GPU
    start = time.time()
    batch_sims = gpu_sim.lorentz_similarity_batch_gpu(U, V)
    gpu_time = time.time() - start

    print(f"Batch GPU processing: {gpu_time*1000:.2f} ms")
    print(f"Result shape: {batch_sims.shape}")
    print(f"Sample similarities: {batch_sims[:5]}")
    print()

    # Compare with sequential CPU processing
    print("Comparison with sequential CPU processing:")
    start = time.time()
    cpu_sims = np.array([lorentz_similarity(U[i], V[i]) for i in range(batch_size)])
    cpu_time = time.time() - start

    print(f"Sequential CPU: {cpu_time*1000:.2f} ms")
    print(f"Speedup: {cpu_time/gpu_time:.2f}x")
    print()

    # Verify consistency
    max_diff = np.max(np.abs(batch_sims - cpu_sims))
    print(f"Maximum difference GPU vs CPU: {max_diff:.2e}")
    print("✓ Results match!" if max_diff < 1e-9 else "✗ Results differ!")


def example_attention_mechanism():
    """Demonstrate GPU acceleration for attention mechanisms."""
    print_section("Example 3: Attention Mechanism Simulation")

    num_tokens = 100
    embedding_dim = 256

    print(f"Simulating self-attention for {num_tokens} tokens")
    print(f"Embedding dimension: {embedding_dim}")
    print()

    # Generate token embeddings
    np.random.seed(123)
    embeddings = np.random.randn(num_tokens, embedding_dim)

    # Compute attention scores using GPU
    start = time.time()
    attention_scores = gpu_sim.lorentz_similarity_matrix_gpu(
        embeddings, embeddings
    )
    gpu_time = time.time() - start

    print(f"GPU attention matrix computation: {gpu_time*1000:.2f} ms")
    print(f"Attention score matrix shape: {attention_scores.shape}")
    print()

    # Analyze self-attention (diagonal)
    diagonal = np.diag(attention_scores)
    print("Self-attention analysis:")
    print(f"  Diagonal values (self-similarity): mean={np.mean(diagonal):.6f}, "
          f"std={np.std(diagonal):.6f}")
    print(f"  All diagonal ~0.0? {np.allclose(diagonal, 0.0, atol=1e-6)}")
    print()

    # Analyze cross-attention (off-diagonal)
    mask = np.ones_like(attention_scores, dtype=bool)
    np.fill_diagonal(mask, False)
    off_diagonal = attention_scores[mask]

    print("Cross-attention analysis:")
    print(f"  Off-diagonal values: mean={np.mean(off_diagonal):.6f}, "
          f"std={np.std(off_diagonal):.6f}")
    print(f"  Range: [{np.min(off_diagonal):.3f}, {np.max(off_diagonal):.3f}]")
    print()

    # Demonstrate attention weights with softmax
    # Apply softmax to get attention weights
    def softmax(x, axis=-1):
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    attention_weights = softmax(attention_scores, axis=1)

    print("After softmax normalization:")
    print(f"  Self-attention weights (diagonal): mean={np.mean(np.diag(attention_weights)):.6f}")
    print(f"  Uniform distribution weight: {1/num_tokens:.6f}")
    print()
    print("✓ Self-attention is NOT amplified (prevents loops!)")


def example_semantic_search():
    """Demonstrate GPU-accelerated semantic search."""
    print_section("Example 4: Large-Scale Semantic Search")

    num_queries = 50
    num_documents = 5000
    embedding_dim = 384  # Common for sentence transformers

    print(f"Simulating semantic search:")
    print(f"  {num_queries} queries")
    print(f"  {num_documents} documents")
    print(f"  {embedding_dim}-dimensional embeddings")
    print()

    # Generate embeddings
    np.random.seed(456)
    query_embeddings = np.random.randn(num_queries, embedding_dim)
    doc_embeddings = np.random.randn(num_documents, embedding_dim)

    # GPU search
    start = time.time()
    similarity_matrix = gpu_sim.lorentz_similarity_matrix_gpu(
        query_embeddings, doc_embeddings
    )
    gpu_time = time.time() - start

    print(f"GPU similarity computation: {gpu_time*1000:.2f} ms")
    print(f"  ({num_queries * num_documents:,} pairwise similarities)")
    print(f"  Throughput: {(num_queries * num_documents) / gpu_time / 1e6:.2f} M pairs/sec")
    print()

    # Find top-k matches for each query
    k = 5
    print(f"Top-{k} document matches per query:")
    for i in range(min(3, num_queries)):  # Show first 3 queries
        top_k_indices = np.argsort(similarity_matrix[i])[-k:][::-1]
        top_k_scores = similarity_matrix[i][top_k_indices]
        print(f"  Query {i}: docs {top_k_indices} "
              f"(scores: {[f'{s:.3f}' for s in top_k_scores]})")


def example_loop_prevention_demo():
    """Demonstrate loop prevention in iterative systems."""
    print_section("Example 5: Loop Prevention in Iterative Systems")

    print("Simulating an iterative refinement system...")
    print()

    # Initial state vector
    state = np.random.randn(64)
    iterations = 10

    print("Scenario: System that updates based on similarity to previous state")
    print()

    # Standard cosine similarity approach
    print("Standard Cosine Similarity:")
    standard_accumulation = 0.0
    for i in range(iterations):
        self_sim = standard_cosine_similarity(state, state)
        standard_accumulation += self_sim
        print(f"  Iteration {i+1}: self-similarity = {self_sim:.6f}, "
              f"accumulated = {standard_accumulation:.6f}")

    print(f"\n  Total accumulated: {standard_accumulation:.6f}")
    print(f"  Average per iteration: {standard_accumulation/iterations:.6f}")
    print(f"  ⚠ High self-reinforcement - potential for loops!")
    print()

    # Lorentz similarity approach
    print("Lorentz-Invariant Similarity:")
    lorentz_accumulation = 0.0
    for i in range(iterations):
        self_sim = gpu_sim.lorentz_similarity_gpu(state, state)
        lorentz_accumulation += self_sim
        print(f"  Iteration {i+1}: self-similarity = {self_sim:.6f}, "
              f"accumulated = {lorentz_accumulation:.6f}")

    print(f"\n  Total accumulated: {lorentz_accumulation:.6f}")
    print(f"  Average per iteration: {lorentz_accumulation/iterations:.6f}")
    print(f"  ✓ Neutral self-reference - loop prevention!")


def performance_comparison():
    """Compare GPU vs CPU performance across different problem sizes."""
    print_section("Example 6: Performance Comparison")

    if not gpu_sim.is_gpu_available():
        print("⚠ GPU not available - skipping performance comparison")
        print("  Install CuPy to enable GPU acceleration")
        return

    print("Comparing GPU vs CPU performance for various problem sizes:")
    print()

    # Test configurations: (batch_size, dimension)
    configs = [
        (100, 64, "Small (100 × 64)"),
        (1000, 128, "Medium (1000 × 128)"),
        (5000, 256, "Large (5000 × 256)"),
    ]

    results = []

    for batch_size, dim, label in configs:
        print(f"\n{label}:")

        # Generate data
        U = np.random.randn(batch_size, dim)
        V = np.random.randn(batch_size, dim)

        # GPU batch processing
        start = time.time()
        _ = gpu_sim.lorentz_similarity_batch_gpu(U, V)
        gpu_time = time.time() - start

        # CPU sequential processing
        start = time.time()
        _ = np.array([lorentz_similarity(U[i], V[i]) for i in range(batch_size)])
        cpu_time = time.time() - start

        speedup = cpu_time / gpu_time

        print(f"  GPU: {gpu_time*1000:8.2f} ms")
        print(f"  CPU: {cpu_time*1000:8.2f} ms")
        print(f"  Speedup: {speedup:6.2f}x")

        results.append((label, batch_size * dim, gpu_time, cpu_time, speedup))

    print("\n" + "-" * 70)
    print("Summary:")
    print(f"{'Configuration':<20} {'Problem Size':<15} {'GPU (ms)':<12} "
          f"{'CPU (ms)':<12} {'Speedup':<10}")
    print("-" * 70)
    for label, size, gpu_t, cpu_t, speedup in results:
        print(f"{label:<20} {size:<15,} {gpu_t*1000:<12.2f} "
              f"{cpu_t*1000:<12.2f} {speedup:<10.2f}x")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("  GPU-Accelerated Lorentz-Invariant Similarity Examples")
    print("  EigenFunction: Loop Prevention via Lightlike Boundary")
    print("=" * 70)

    check_gpu_status()

    example_basic_usage()
    example_batch_processing()
    example_attention_mechanism()
    example_semantic_search()
    example_loop_prevention_demo()
    performance_comparison()

    print("\n" + "=" * 70)
    print("  All examples completed!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
