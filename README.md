# EigenFunction: Lorentz-Invariant Similarity for Loop Prevention

A novel approach to similarity measures in self-referential systems that uses Lorentz-invariant geometry to prevent pathological feedback loops.

## Overview

This repository implements a **Lorentz-invariant cosine similarity measure** that addresses a fundamental problem in recursive and iterative systems: self-reinforcing loops caused by perfect self-similarity.

### The Problem

Standard cosine similarity yields **self-similarity = 1.0**, which can exacerbate infinite loops in:
- Recursive attention mechanisms (transformers)
- Graph traversal algorithms
- Reinforcement learning loops
- Iterative refinement systems
- Semantic search with query expansion
- Consciousness modeling frameworks

### The Solution

By computing similarity on the **lightlike boundary** (ds² = 0) using Lorentz-invariant geometry, we obtain **self-similarity = 0.0**. This neutral value:

1. **Disrupts self-reinforcement** - prevents systems from over-weighting their current state
2. **Promotes evolution** - forces incorporation of external information
3. **Prevents collapse** - avoids fixed-point attractors in state space
4. **Aligns with eigengate principles** - measurements inherently change the system

## Theoretical Foundation

### Mathematical Formulation

For vectors **u** and **v** in n-dimensional space, we embed them in (n+1)-dimensional Minkowski spacetime:

- **u** → (**u**, ||**u**||)
- **v** → (**v**, ||**v**||)

The Lorentz inner product is:

```
⟨u, v⟩_L = u·v - ||u|| * ||v||
```

For self-similarity:

```
⟨u, u⟩_L = ||u||² - ||u||² = 0
```

This is the **lightlike** (null) condition from special relativity, where ds² = 0.

### Connection to Eigengate Framework

The lightlike self-similarity aligns with eigengate principles for consciousness modeling:

- **Observation disrupts self-reference**: Measuring a system on the lightlike boundary inherently changes it
- **No fixed ontological states**: Prevents collapse to permanent configurations
- **Evolutionary continuity**: Promotes ongoing state evolution rather than stagnation
- **Process over substance**: Aligns with process philosophy's "no permanent self"

### Scope and Limitations

**This is NOT:**
- A solution to the halting problem (which is undecidable in general)
- A universal loop detector
- Applicable to arbitrary Turing-computable programs

**This IS:**
- A geometric safeguard for specifically designed architectures
- A mechanism to reduce loop propensity in similarity-based algorithms
- A theoretical framework connecting relativity to computational stability

## Installation

```bash
git clone https://github.com/InauguralPhysicist/EigenFunction.git
cd EigenFunction
pip install -r requirements.txt
```

### Dependencies

**Core Dependencies:**
- NumPy >= 1.20.0
- pytest >= 7.0.0 (for testing)

**Optional (for GPU acceleration):**
- CuPy (CUDA 11.x or 12.x)
  - For CUDA 11.x: `pip install cupy-cuda11x`
  - For CUDA 12.x: `pip install cupy-cuda12x`
  - GPU module works without CuPy, automatically falling back to CPU

**Optional (for PyTorch neural network modules):**
- PyTorch >= 2.0
  - CPU only: `pip install torch`
  - With CUDA: See [pytorch.org](https://pytorch.org/get-started/locally/)
  - Required for `eigen_attention.py` and `eigen_memory.py` modules

## Usage

### Basic Similarity Computation

```python
import numpy as np
from similarity import lorentz_similarity, standard_cosine_similarity

# Create vectors
u = np.array([1.0, 2.0, 3.0])
v = np.array([4.0, 5.0, 6.0])

# Standard cosine similarity
standard_sim = standard_cosine_similarity(u, v)
print(f"Standard similarity: {standard_sim}")

# Lorentz-invariant similarity
lorentz_sim = lorentz_similarity(u, v)
print(f"Lorentz similarity: {lorentz_sim}")
```

### Self-Similarity Comparison

```python
from similarity import compare_self_similarity

v = np.array([3.0, 4.0])
result = compare_self_similarity(v)

print(f"Standard self-similarity: {result['standard']}")  # 1.000000
print(f"Lorentz self-similarity: {result['lorentz']}")    # 0.000000
```

### Loop Prevention in Attention Mechanisms

```python
# Token embeddings in a transformer-style attention layer
query = np.array([0.8, 0.6, 0.3])
keys = [
    np.array([0.8, 0.6, 0.3]),  # Same as query (self)
    np.array([0.2, 0.9, 0.1]),  # Different token
    np.array([0.5, 0.5, 0.5]),  # Different token
]

# Standard approach: self-attention dominates
standard_weights = [standard_cosine_similarity(query, k) for k in keys]
# [1.0, 0.xx, 0.yy] - self gets maximum weight!

# Lorentz approach: self-attention neutralized
lorentz_weights = [lorentz_similarity(query, k) for k in keys]
# [0.0, 0.xx, 0.yy] - self gets neutral weight
```

### GPU-Accelerated Computation

For large-scale applications, use the GPU module for significant speedup:

```python
import numpy as np
import gpu_similarity as gpu_sim

# Check GPU availability
if gpu_sim.is_gpu_available():
    print("GPU acceleration enabled!")
else:
    print("Using CPU fallback")

# Single vector pair (automatically uses GPU if available)
u = np.random.randn(512)
v = np.random.randn(512)
sim = gpu_sim.lorentz_similarity_gpu(u, v)

# Batch processing (10-100x faster on GPU)
U = np.random.randn(1000, 512)  # 1000 vectors
V = np.random.randn(1000, 512)
similarities = gpu_sim.lorentz_similarity_batch_gpu(U, V)  # Shape: (1000,)

# Attention mechanism (pairwise similarity matrix)
embeddings = np.random.randn(100, 256)  # 100 tokens
attention_scores = gpu_sim.lorentz_similarity_matrix_gpu(embeddings, embeddings)
# Shape: (100, 100), diagonal elements are ~0.0 (loop prevention!)

# Automatic GPU/CPU selection
sim = gpu_sim.lorentz_similarity_auto(u, v, prefer_gpu=True)
```

### PyTorch Neural Network Modules

EigenFunction provides ready-to-use PyTorch modules for attention and memory:

```python
import torch
from eigen_attention import EigenAttention
from eigen_memory import EigenMemory

# EigenAttention: Multi-head attention with loop prevention
attn = EigenAttention(
    dim=512,              # Model dimension
    num_heads=8,          # Number of attention heads
    sim_scale=4.0,        # Scale factor for similarities
    loop_epsilon=1e-3,    # Loop prevention threshold
    causal=False          # Set True for autoregressive models
).to('cuda')

# Process sequences
x = torch.randn(32, 128, 512).cuda()  # (batch, seq_len, dim)
output, attn_weights = attn(x)
# output: (32, 128, 512), attn_weights: (32, 8, 128, 128)
# Diagonal of attn_weights is suppressed (loop prevention!)

# EigenMemory: Differentiable memory with Lorentz-based retrieval
memory = EigenMemory(
    dim=512,              # Memory vector dimension
    max_mem_slots=4096,   # Maximum memory capacity
    k_top=32,             # Top-k neighbors to retrieve
    loop_epsilon=1e-3,    # Loop prevention threshold
    decay=0.99            # Temporal decay factor
).to('cuda')

# Write to memory
states = torch.randn(100, 512).cuda()
memory.write(states)

# Retrieve from memory
queries = torch.randn(32, 512).cuda()
retrieved = memory(queries)  # (32, 512)
# Retrieved vectors won't be identical to queries (loop prevention)

# Retrieve with attention weights
retrieved, (attn, indices) = memory(queries, return_weights=True)
# attn: (32, k_top), indices: (32, k_top)
```

**Key Features:**

- **Gradient Flow**: Both modules are fully differentiable for end-to-end training
- **Loop Prevention**: Self-similarity ~0 prevents pathological feedback
- **Device Support**: Automatically handles CPU/CUDA placement
- **Efficient**: Leverages GPU-accelerated similarity computation
- **Flexible**: Compatible with standard PyTorch training pipelines

**Example: Transformer with Loop Prevention**

```python
import torch.nn as nn

class EigenTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.attn = EigenAttention(dim, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        # Self-attention with loop prevention
        attn_out, _ = self.attn(self.norm1(x))
        x = x + attn_out

        # MLP
        x = x + self.mlp(self.norm2(x))
        return x

# Use in model
model = EigenTransformerBlock(dim=512, num_heads=8).cuda()
x = torch.randn(32, 128, 512).cuda()
output = model(x)
```

## Examples

### CPU Examples

Run the comprehensive demonstration suite:

```bash
python examples_loop_prevention.py
```

This demonstrates loop prevention in:

1. **Self-Attention Mechanisms** - Prevents attention collapse to self-tokens
2. **Graph Traversal** - Eliminates self-loop attractors in semantic networks
3. **Iterative Refinement** - Maintains update momentum in optimization
4. **Semantic Search** - Encourages query expansion exploration
5. **Consciousness Modeling** - Implements eigengate measurement disruption

### GPU Examples

Run the GPU acceleration examples:

```bash
python examples_gpu.py
```

This demonstrates:

1. **GPU Status Check** - Verify CUDA availability and device info
2. **Basic GPU Usage** - Simple GPU-accelerated similarity computation
3. **Batch Processing** - Efficient processing of thousands of vector pairs
4. **Attention Mechanisms** - Large-scale attention score matrices
5. **Semantic Search** - Query-document similarity at scale
6. **Performance Comparison** - GPU vs CPU speedup measurements

## Testing

### CPU Tests

Run the CPU test suite:

```bash
pytest test_similarity.py -v
```

Tests validate:
- ✓ Self-similarity properties (standard = 1.0, Lorentz = 0.0)
- ✓ Orthogonal and parallel vector behavior
- ✓ Numerical stability (zero vectors, extreme magnitudes)
- ✓ High-dimensional performance (up to 500D tested)
- ✓ Loop prevention via accumulation tests
- ✓ Input validation and error handling

### GPU Tests

Run the GPU test suite:

```bash
pytest test_gpu_similarity.py -v
```

GPU tests validate:
- ✓ GPU availability detection and fallback mechanisms
- ✓ Consistency between GPU and CPU implementations
- ✓ Batch processing correctness
- ✓ Similarity matrix computation
- ✓ Attention mechanism simulation
- ✓ Numerical stability on GPU
- ✓ Large-scale performance characteristics

### PyTorch Module Tests

Run the PyTorch module test suite:

```bash
pytest test_eigen_modules.py -v
```

PyTorch tests validate:
- ✓ EigenAttention module functionality and gradient flow
- ✓ EigenMemory module write/read operations
- ✓ Loop prevention in attention (self-attention suppression)
- ✓ Loop prevention in memory (self-retrieval avoidance)
- ✓ Causal masking and external attention masks
- ✓ Multi-head attention independence
- ✓ Device handling (CPU/CUDA)
- ✓ Gradient backpropagation through both modules
- ✓ Integration tests (attention + memory pipelines)
- ✓ Edge cases and error handling

Run all tests (excluding slow tests):

```bash
pytest -v -m "not slow"
```

## Applications

### Recommended Use Cases

1. **Neural Network Attention**
   - Self-attention layers in transformers
   - Cross-attention with potential self-reference
   - Memory-augmented networks with retrieval

2. **Graph Algorithms**
   - Semantic similarity graphs
   - Knowledge graph traversal
   - Community detection with similarity metrics

3. **Reinforcement Learning**
   - State similarity in exploration policies
   - Experience replay with similarity-based sampling
   - Hierarchical RL with state abstraction

4. **Information Retrieval**
   - Query expansion with relevance feedback
   - Semantic search with iterative refinement
   - Document clustering with self-organization

5. **Theoretical Modeling**
   - Consciousness and self-reference (eigengate framework)
   - Adaptive control systems
   - Evolutionary algorithms with similarity-based selection

### When NOT to Use

- Simple one-off similarity comparisons without recursion
- Systems where self-similarity = 1.0 is semantically meaningful
- Applications requiring strict cosine similarity properties
- Performance-critical inner loops (Lorentz requires additional computation)

## Performance Characteristics

### CPU Implementation

- **Time Complexity**: O(n) where n is vector dimension (same as standard cosine)
- **Space Complexity**: O(1) beyond input storage
- **Numerical Stability**: Tested with vectors from 1e-15 to 1e15 magnitude
- **Precision**: Uses float64 throughout with epsilon = 1e-10 for stability

### GPU Implementation

- **Batch Processing**: 10-100x speedup over sequential CPU for batches > 1000
- **Matrix Operations**: Highly optimized for attention mechanisms (N×M similarity matrices)
- **Memory**: Automatically manages GPU memory transfers
- **Fallback**: Gracefully falls back to CPU when GPU unavailable
- **Supported Operations**:
  - Single pair: `lorentz_similarity_gpu(u, v)`
  - Batch pairs: `lorentz_similarity_batch_gpu(U, V)` - shape (N,D) → (N,)
  - Pairwise matrix: `lorentz_similarity_matrix_gpu(U, V)` - shape (N,D) × (M,D) → (N,M)

**Performance Example** (NVIDIA GPU, 1000 vectors × 128 dimensions):
- GPU batch: ~5 ms
- CPU sequential: ~200 ms
- Speedup: ~40x

## Theoretical Connections

### Physics

- **Special Relativity**: Lightlike worldlines (ds² = 0) for massless particles
- **Causal Structure**: Null boundaries in spacetime geometry
- **Invariance**: Preserved under Lorentz transformations

### Mathematics

- **Minkowski Space**: Pseudo-Riemannian manifold with signature (-,+,+,+)
- **Null Vectors**: Self-orthogonal elements in indefinite metric spaces
- **Eigenfunction Theory**: Fixed points and spectral properties

### Philosophy

- **Process Philosophy**: No permanent substantial self
- **Phenomenology**: Observer-observed entanglement
- **Eastern Philosophy**: "No-self" (anatta) doctrine alignment

## Future Work

- [x] ~~GPU-accelerated implementation for large-scale applications~~ **✓ Completed**
- [x] ~~PyTorch integration~~ **✓ Completed**
  - ✓ EigenAttention: Multi-head attention with loop prevention
  - ✓ EigenMemory: Differentiable memory module with Lorentz retrieval
  - ✓ Full gradient support for end-to-end training
  - ✓ Comprehensive test suite
- [ ] TensorFlow integration
  - Custom TensorFlow operations for batch processing
  - Keras layers for attention and memory
- [ ] Empirical validation on real-world datasets
  - Transformer models with Lorentz attention
  - Graph neural networks with loop prevention
  - Reinforcement learning with state similarity
- [ ] Extension to other similarity measures (Jaccard, Euclidean)
- [ ] Theoretical analysis of convergence properties
- [ ] Application to specific consciousness modeling architectures
- [ ] Multi-GPU support for massive-scale computations
- [ ] Sparse matrix optimizations for very large graphs

## Contributing

Contributions welcome! Areas of interest:

- Empirical validation studies
- Application to new domains
- Performance optimizations
- Theoretical extensions
- Bug fixes and documentation improvements

## License

MIT License - see LICENSE file for details

## Citation

If you use this work in research, please cite:

```
@software{eigenfunction2025,
  author = {InauguralPhysicist},
  title = {EigenFunction: Lorentz-Invariant Similarity for Loop Prevention},
  year = {2025},
  url = {https://github.com/InauguralPhysicist/EigenFunction}
}
```

## Contact

- **Author**: InauguralPhysicist
- **Email**: jon92087@yahoo.com
- **Related Work**: [eigenai.org](https://eigenai.org)

## Acknowledgments

This work builds on ideas from:
- Relativistic physics and Minkowski geometry
- Eigengate consciousness framework
- Information theory and self-reference
- Process philosophy and evolutionary epistemology

---

**Note**: This is a research prototype exploring novel theoretical connections. Empirical validation for production use is recommended.
