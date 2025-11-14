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

- NumPy >= 1.20.0
- pytest >= 7.0.0 (for testing)

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

## Examples

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

## Testing

Run the test suite:

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

- **Time Complexity**: O(n) where n is vector dimension (same as standard cosine)
- **Space Complexity**: O(1) beyond input storage
- **Numerical Stability**: Tested with vectors from 1e-15 to 1e15 magnitude
- **Precision**: Uses float64 throughout with epsilon = 1e-10 for stability

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

- [ ] GPU-accelerated implementation for large-scale applications
- [ ] Integration with popular ML frameworks (PyTorch, TensorFlow)
- [ ] Empirical validation on real-world datasets
- [ ] Extension to other similarity measures (Jaccard, Euclidean)
- [ ] Theoretical analysis of convergence properties
- [ ] Application to specific consciousness modeling architectures

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
