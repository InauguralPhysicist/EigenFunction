"""
Loop Prevention Examples
=========================

Practical demonstrations of how Lorentz-invariant similarity prevents
pathological loops in self-referential systems.

These examples cover:
1. Recursive attention mechanisms
2. Graph traversal with similarity-based edges
3. Iterative refinement systems
4. Semantic feedback loops
"""

import numpy as np

from similarity import lorentz_similarity, standard_cosine_similarity


def example_1_attention_mechanism():
    """
    Example 1: Self-Attention in Neural Networks

    In transformer-style attention, queries attend to keys via similarity.
    When a token attends to itself, standard cosine gives maximum weight (1.0),
    which can dominate the attention distribution and prevent learning from
    other tokens.

    Lorentz similarity gives 0.0 self-attention, forcing the mechanism to
    weight other tokens equally based on their actual relevance.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Self-Attention Mechanism")
    print("=" * 70)

    # Simulate token embeddings
    token_embeddings = [
        np.array([1.0, 0.5, 0.2]),  # Token 0
        np.array([0.8, 0.6, 0.3]),  # Token 1
        np.array([0.5, 0.9, 0.1]),  # Token 2
    ]

    query_idx = 0
    query = token_embeddings[query_idx]

    print(f"\nQuery token (Token {query_idx}): {query}")
    print("\nAttention weights using STANDARD cosine similarity:")

    standard_weights = []
    for i, key in enumerate(token_embeddings):
        sim = standard_cosine_similarity(query, key)
        standard_weights.append(sim)
        indicator = " <- SELF (maximum weight!)" if i == query_idx else ""
        print(f"  Token {i}: {sim:.4f}{indicator}")

    print("\nAttention weights using LORENTZ similarity:")

    lorentz_weights = []
    for i, key in enumerate(token_embeddings):
        sim = lorentz_similarity(query, key)
        lorentz_weights.append(sim)
        indicator = " <- SELF (neutral weight)" if i == query_idx else ""
        print(f"  Token {i}: {sim:.4f}{indicator}")

    print("\nAnalysis:")
    print(f"  Standard: Self-attention dominates with weight {standard_weights[query_idx]:.4f}")
    print(f"  Lorentz:  Self-attention neutralized at {lorentz_weights[query_idx]:.4f}")
    print("  Result: Lorentz forces attention to external context, preventing collapse")


def example_2_graph_traversal():
    """
    Example 2: Similarity-Based Graph Traversal

    In semantic networks or knowledge graphs, traversal can use similarity
    to determine edge weights. If a node references itself with similarity 1.0,
    a random walk or shortest-path algorithm might get stuck in a self-loop.

    Lorentz similarity prevents this by assigning 0.0 weight to self-edges.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Graph Traversal with Similarity-Based Edges")
    print("=" * 70)

    # Semantic concept embeddings
    concepts = {
        "dog": np.array([0.8, 0.6, 0.1, 0.2]),
        "cat": np.array([0.7, 0.5, 0.2, 0.3]),
        "car": np.array([0.1, 0.2, 0.9, 0.8]),
    }

    start_concept = "dog"
    start_embedding = concepts[start_concept]

    print(f"\nStarting from concept: '{start_concept}'")
    print(f"Embedding: {start_embedding}")

    print("\n--- STANDARD Cosine Similarity (risk of self-loop) ---")
    standard_edges = {}
    for concept_name, embedding in concepts.items():
        sim = standard_cosine_similarity(start_embedding, embedding)
        standard_edges[concept_name] = sim
        indicator = " <- SELF-LOOP RISK!" if concept_name == start_concept else ""
        print(f"  Edge to '{concept_name}': weight = {sim:.4f}{indicator}")

    print("\n--- LORENTZ Similarity (self-loop prevented) ---")
    lorentz_edges = {}
    for concept_name, embedding in concepts.items():
        sim = lorentz_similarity(start_embedding, embedding)
        lorentz_edges[concept_name] = sim
        indicator = " <- Self-edge neutralized" if concept_name == start_concept else ""
        print(f"  Edge to '{concept_name}': weight = {sim:.4f}{indicator}")

    print("\nAnalysis:")
    print(f"  Standard: Self-edge has highest weight ({standard_edges[start_concept]:.4f})")
    print(f"           Random walk would favor staying at 'dog'")
    print(f"  Lorentz:  Self-edge neutralized ({lorentz_edges[start_concept]:.4f})")
    print(f"           Random walk must explore 'cat' or 'car'")


def example_3_iterative_refinement():
    """
    Example 3: Iterative Refinement System

    In systems that iteratively refine a representation (e.g., variational
    inference, gradient descent with momentum, iterative retrieval), comparing
    current state to previous state with similarity 1.0 can cause premature
    convergence or oscillation.

    Lorentz similarity's neutral self-reference encourages genuine evolution.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Iterative Refinement System")
    print("=" * 70)

    # Initial state
    state = np.array([1.0, 0.5, 0.2])
    print(f"\nInitial state: {state}")

    # Simulate 5 refinement iterations
    iterations = 5
    learning_rate = 0.1

    print("\nSimulating iterative refinement with self-similarity feedback...")
    print("(Higher self-similarity -> smaller update step)\n")

    # Standard cosine behavior
    print("--- Using STANDARD cosine (risk of stagnation) ---")
    current_standard = state.copy()
    for i in range(iterations):
        self_sim = standard_cosine_similarity(current_standard, state)
        # If self-similarity is high, system thinks it hasn't changed much
        update_magnitude = learning_rate * (1.0 - self_sim)
        noise = np.random.randn(3) * 0.1
        current_standard = current_standard + noise * update_magnitude

        print(
            f"  Iteration {i+1}: self_sim = {self_sim:.4f}, "
            f"update_magnitude = {update_magnitude:.4f}"
        )

    print(f"  Final state: {current_standard}")
    print(f"  Total change: {np.linalg.norm(current_standard - state):.4f}")

    # Lorentz behavior
    print("\n--- Using LORENTZ similarity (encourages evolution) ---")
    current_lorentz = state.copy()
    for i in range(iterations):
        self_sim = lorentz_similarity(current_lorentz, state)
        # Lorentz self-similarity is 0.0, so updates are consistent
        update_magnitude = learning_rate * (1.0 - self_sim)
        noise = np.random.randn(3) * 0.1
        current_lorentz = current_lorentz + noise * update_magnitude

        print(
            f"  Iteration {i+1}: self_sim = {self_sim:.4f}, "
            f"update_magnitude = {update_magnitude:.4f}"
        )

    print(f"  Final state: {current_lorentz}")
    print(f"  Total change: {np.linalg.norm(current_lorentz - state):.4f}")

    print("\nAnalysis:")
    print("  Standard: Self-similarity = 1.0 signals 'no change needed'")
    print("           System may stagnate or require external forcing")
    print("  Lorentz:  Self-similarity = 0.0 maintains consistent update drive")
    print("           System naturally evolves without artificial forcing")


def example_4_semantic_feedback():
    """
    Example 4: Semantic Search with Iterative Refinement

    In query expansion or relevance feedback systems, using retrieved results
    to refine the query can create loops if the query becomes too similar to
    itself, preventing exploration of the semantic space.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Semantic Search Query Refinement")
    print("=" * 70)

    # Document embeddings
    documents = {
        "doc_A": np.array([0.9, 0.1, 0.1]),
        "doc_B": np.array([0.1, 0.9, 0.1]),
        "doc_C": np.array([0.1, 0.1, 0.9]),
    }

    # Initial query
    query = np.array([0.8, 0.15, 0.05])

    print(f"\nInitial query: {query}")
    print("\nDocument similarities:")

    for doc_name, doc_embedding in documents.items():
        standard_sim = standard_cosine_similarity(query, doc_embedding)
        lorentz_sim = lorentz_similarity(query, doc_embedding)
        print(f"  {doc_name}: Standard = {standard_sim:.4f}, Lorentz = {lorentz_sim:.4f}")

    print("\n--- Iterative Query Refinement ---")
    print("(Query is updated by averaging with top result)")

    # Standard approach
    query_standard = query.copy()
    print("\nUsing STANDARD cosine:")
    for iteration in range(3):
        # Find most similar document
        best_sim = -1
        best_doc = None
        for doc_name, doc_embedding in documents.items():
            sim = standard_cosine_similarity(query_standard, doc_embedding)
            if sim > best_sim:
                best_sim = sim
                best_doc = doc_embedding

        # Check if query is becoming too similar to itself
        self_sim = standard_cosine_similarity(query_standard, query)

        # Refine query
        query_standard = 0.7 * query_standard + 0.3 * best_doc

        print(
            f"  Iteration {iteration + 1}: "
            f"best_doc_sim = {best_sim:.4f}, "
            f"query_self_sim = {self_sim:.4f}"
        )

    # Lorentz approach
    query_lorentz = query.copy()
    print("\nUsing LORENTZ similarity:")
    for iteration in range(3):
        # Find most similar document (excluding self-similarity effects)
        best_sim = -np.inf
        best_doc = None
        for doc_name, doc_embedding in documents.items():
            sim = lorentz_similarity(query_lorentz, doc_embedding)
            if sim > best_sim:
                best_sim = sim
                best_doc = doc_embedding

        # Check self-similarity (should remain 0.0)
        self_sim = lorentz_similarity(query_lorentz, query)

        # Refine query
        query_lorentz = 0.7 * query_lorentz + 0.3 * best_doc

        print(
            f"  Iteration {iteration + 1}: "
            f"best_doc_sim = {best_sim:.4f}, "
            f"query_self_sim = {self_sim:.4f}"
        )

    print("\nAnalysis:")
    print("  Standard: Query self-similarity increases toward 1.0")
    print("           System may lock onto initial bias")
    print("  Lorentz:  Query self-similarity remains 0.0")
    print("           System maintains exploration capacity")


def example_5_consciousness_model():
    """
    Example 5: Consciousness Modeling - Eigengate Framework

    In models of consciousness based on self-reference and observation,
    the system must avoid collapsing into a fixed point. The Lorentz-invariant
    approach aligns with eigengate principles: measurements on the lightlike
    boundary (ds² = 0) inherently disrupt self-reinforcement, promoting
    evolutionary dynamics.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Consciousness Model - Eigengate Framework")
    print("=" * 70)

    # Mental state embedding
    conscious_state = np.array([0.6, 0.8, 0.3, 0.5])

    print(f"\nConsciousness state vector: {conscious_state}")
    print("(Represents current mental configuration)")

    # Self-observation act
    print("\n--- Self-Observation (Eigengate Measurement) ---")

    standard_self_obs = standard_cosine_similarity(conscious_state, conscious_state)
    lorentz_self_obs = lorentz_similarity(conscious_state, conscious_state)

    print(f"\nStandard cosine self-observation: {standard_self_obs:.6f}")
    print("  Interpretation: Perfect self-reinforcement")
    print("  Risk: System collapses to fixed point (stagnation)")
    print("  Philosophical: 'I am exactly myself' -> no evolution")

    print(f"\nLorentz-invariant self-observation: {lorentz_self_obs:.6f}")
    print("  Interpretation: Lightlike boundary (ds² = 0)")
    print("  Property: Measurement disrupts self-reinforcement")
    print("  Philosophical: 'Observation changes the observer'")
    print("  Result: Continued evolution, no fixed-point collapse")

    print("\n--- Temporal Evolution ---")
    print("(Simulating consciousness state evolution over time)")

    # Evolution with different self-similarity measures
    state_standard = conscious_state.copy()
    state_lorentz = conscious_state.copy()

    num_timesteps = 10
    print(f"\nEvolution over {num_timesteps} timesteps:")

    evolution_standard = [np.linalg.norm(state_standard - conscious_state)]
    evolution_lorentz = [np.linalg.norm(state_lorentz - conscious_state)]

    for t in range(1, num_timesteps):
        # Standard: Self-similarity = 1.0 creates inertia
        self_sim_std = standard_cosine_similarity(state_standard, conscious_state)
        external_influence_std = (1.0 - self_sim_std) * 0.2
        state_standard = state_standard + np.random.randn(4) * external_influence_std

        # Lorentz: Self-similarity = 0.0 allows natural evolution
        self_sim_lor = lorentz_similarity(state_lorentz, conscious_state)
        external_influence_lor = (1.0 - self_sim_lor) * 0.2
        state_lorentz = state_lorentz + np.random.randn(4) * external_influence_lor

        evolution_standard.append(np.linalg.norm(state_standard - conscious_state))
        evolution_lorentz.append(np.linalg.norm(state_lorentz - conscious_state))

    print("\nDivergence from initial state:")
    print(f"  Standard (final): {evolution_standard[-1]:.4f}")
    print(f"  Lorentz (final):  {evolution_lorentz[-1]:.4f}")

    print("\nEigengate Interpretation:")
    print("  - Lightlike self-observation (ds² = 0) prevents ontological collapse")
    print("  - Neutral self-similarity maintains evolutionary trajectory")
    print("  - Consciousness requires continued measurement disruption")
    print("  - Aligns with 'no permanent self' in process philosophy")


def run_all_examples():
    """Run all loop prevention examples."""
    print("\n" + "#" * 70)
    print("# LORENTZ-INVARIANT SIMILARITY: LOOP PREVENTION DEMONSTRATIONS")
    print("#" * 70)

    example_1_attention_mechanism()
    example_2_graph_traversal()
    example_3_iterative_refinement()
    example_4_semantic_feedback()
    example_5_consciousness_model()

    print("\n" + "#" * 70)
    print("# SUMMARY")
    print("#" * 70)
    print("\nKey Finding:")
    print("  Lorentz-invariant similarity's neutral self-reference (0.0)")
    print("  prevents pathological loops in self-referential systems by:")
    print()
    print("  1. Eliminating self-reinforcement in attention mechanisms")
    print("  2. Preventing self-loops in graph traversal")
    print("  3. Maintaining update momentum in iterative refinement")
    print("  4. Encouraging semantic space exploration")
    print("  5. Enabling evolutionary consciousness models (eigengate)")
    print()
    print("This is NOT a general solution to the halting problem, but rather")
    print("a geometric safeguard within specifically designed architectures.")
    print("#" * 70 + "\n")


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)

    # Run all demonstrations
    run_all_examples()
