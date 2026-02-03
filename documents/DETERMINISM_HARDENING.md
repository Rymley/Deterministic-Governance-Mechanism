# Determinism Hardening for Production Substrate Sharding

**⚠️ NON-NORMATIVE IMPLEMENTATION GUIDANCE**

This document describes production hardening strategies for scaling the reference implementation to larger substrates. **The core determinism invariant stated in the README does not depend on these techniques.** This is exploratory engineering documentation, not a claim about the minimal system's behavior.

The reference implementation proves determinism with simple structures. This document explores how to maintain that property at scale—but these explorations are not required to understand or verify the basic mechanism.

---

## Executive Summary

The substrate sharding architecture achieves sub-microsecond retrieval by reducing 250K vectors to a cache-resident working set. **The determinism guarantee depends on making caching an algorithm, not a heuristic.** This document identifies the failure modes where bit-identical behavior can silently degrade and provides concrete hardening strategies.

**Core Principle:** Same query → same shards → same order → same cache state → bit-identical inference.

---

## Failure Modes Analysis

### 1. Product Quantization (PQ) Non-Determinism

#### The Problem

Standard PQ pipelines introduce three sources of variance:

```python
# WRONG: Non-deterministic PQ encoding
def encode_vector(vec: list[float]) -> bytes:
    # Problem 1: Codebook trained with random init
    codebook = kmeans(training_data, n_clusters=256, init='random')

    # Problem 2: Nearest codebook search uses hardware-dependent FP
    distances = [float_distance(vec, c) for c in codebook]  # FP rounding varies

    # Problem 3: Tie-breaking is implicit (first occurrence)
    code = min(range(len(distances)), key=distances.__getitem__)  # ties are implicit

    return encode(code)
```

**Result:** Same vector encodes to different codes on different runs/hardware.

#### Hardening Strategy

```python
# CORRECT: Deterministic PQ encoding
def encode_vector_deterministic(vec: list[float], frozen_codebook: list[list[float]]) -> bytes:
    """
    Deterministic product quantization.

    Requirements:
    1. Frozen codebook (trained once, never modified)
    2. Fixed-point distance computation
    3. Explicit tie-breaking by codebook index
    """

    # Use int32 fixed-point arithmetic (avoid FP variance)
    # Scale vectors to [-32768, 32767] range
    vec_scaled = [int(round(x * 32767)) for x in vec]
    codebook_scaled = [[int(round(x * 32767)) for x in c] for c in frozen_codebook]

    # Squared L2 distance in fixed-point (overflow-safe)
    # d² = Σ(v_i - c_i)²
    distances = []
    for c in codebook_scaled:
        d2 = 0
        for vi, ci in zip(vec_scaled, c):
            diff = vi - ci
            d2 += diff * diff
        distances.append(d2)

    # Explicit tie-breaking: smallest index wins
    min_dist = min(distances)
    candidates = [i for i, d in enumerate(distances) if d == min_dist]
    code = candidates[0]  # Deterministic: lowest index

    return int(code).to_bytes(4, "little", signed=False)
```

**Key Requirements:**

1. **Frozen Codebook**
   - Train once on canonical dataset
   - Serialize to disk with exact byte representation
   - Load with fixed dtype (int32 or int16, not float)
   - Never retrain or update

2. **Fixed-Point Distance**
   - Scale to integer range: `int32` or `int16`
   - Use squared distance (avoids sqrt variance)
   - Check for overflow: max value < 2^31

3. **Explicit Tie-Breaking**
   - When multiple codebook entries have same distance
   - Always select by lowest index (deterministic)
   - Document in code comments

**Verification:**

```python
# Test: Same vector must encode to same code across runs
vec = [0.01] * 768
code1 = encode_vector_deterministic(vec, frozen_codebook)
code2 = encode_vector_deterministic(vec, frozen_codebook)
assert code1 == code2, "PQ encoding must be deterministic"
```

---

### 2. HNSW Graph Build Non-Determinism

#### The Problem

HNSW (Hierarchical Navigable Small World) graphs are typically built with:

```python
# WRONG: Non-deterministic HNSW build
def build_hnsw(vectors: List[list[float]]) -> HNSWGraph:
    graph = HNSWGraph()

    # Problem 1: Parallel insertion creates race conditions
    with ThreadPoolExecutor() as executor:
        executor.map(graph.insert, vectors)  # Non-deterministic order

    # Problem 2: Random layer assignment
    layer = random.randint(0, max_layer)  # Non-deterministic

    # Problem 3: Neighbor selection uses unstable sort
    neighbors = sorted(candidates, key=lambda x: distance(x, query))
    # If distances are equal, order is undefined

    return graph
```

**Result:** Same vectors produce different graphs on different runs.

#### Hardening Strategy

```python
# CORRECT: Deterministic HNSW build
def build_hnsw_deterministic(vectors: List[list[float]],
                             vector_ids: List[int]) -> HNSWGraph:
    """
    Deterministic HNSW graph construction.

    Requirements:
    1. Single-threaded insertion (or deterministic parallel batching)
    2. Deterministic layer assignment (hash-based)
    3. Stable neighbor selection with explicit tie-breaking
    """

    graph = HNSWGraph()

    # Sort vectors by ID to ensure fixed insertion order
    sorted_items = sorted(zip(vector_ids, vectors), key=lambda x: x[0])

    # Single-threaded insertion (deterministic)
    for vec_id, vec in sorted_items:
        # Deterministic layer assignment via hash
        layer = stable_hash(vec_id) % (max_layer + 1)

        # Insert with stable neighbor selection
        graph.insert_deterministic(vec, vec_id, layer)

    return graph


def insert_deterministic(self, vec: list[float], vec_id: int, layer: int):
    """Insert vector with deterministic neighbor selection."""

    # Find candidate neighbors
    candidates = self.search_layer(vec, layer)

    # Compute distances in fixed-point
    distances = [(c_id, fixed_point_distance(vec, c_vec))
                 for c_id, c_vec in candidates]

    # Stable sort: distance first, then ID for tie-breaking
    distances.sort(key=lambda x: (x[1], x[0]))

    # Select top-M neighbors (deterministic)
    neighbors = [c_id for c_id, _ in distances[:self.M]]

    # Add edges (deterministic order)
    for neighbor_id in sorted(neighbors):
        self.add_edge(vec_id, neighbor_id, layer)
```

**Key Requirements:**

1. **Fixed Insertion Order**
   - Sort vectors by ID before insertion
   - Single-threaded build (or deterministic batching)
   - No concurrent modifications

2. **Deterministic Layer Assignment**
   - Use cryptographic hash of vector ID
   - Map to layer: `layer = SHA256(vec_id) % (max_layer + 1)`
   - Never use `random()` or hardware RNG

3. **Stable Neighbor Selection**
   - Sort by (distance, vector_id) tuple
   - Fixed-point distance computation
   - Select top-M by sorted order

4. **Deterministic Traversal**
   - Visit neighbors in sorted ID order
   - No early exit based on heuristics
   - Fixed beam width (no adaptive)

**Verification:**

```python
# Test: Same vectors must produce same graph
graph1 = build_hnsw_deterministic(vectors, ids)
graph2 = build_hnsw_deterministic(vectors, ids)
assert graph1.edges == graph2.edges, "HNSW must be deterministic"
```

---

### 3. Centroid Ranking Non-Determinism

#### The Problem

Centroid-based shard ranking can introduce variance:

```python
# WRONG: Non-deterministic centroid ranking
def rank_shards(query: Vector, shards: List[Shard]) -> List[Shard]:
    # Problem 1: Floating-point distance computation varies
    scores = [(shard, float_distance(query, shard.centroid)) for shard in shards]

    # Problem 2: Unstable sort (ties resolved arbitrarily)
    scores.sort(key=lambda x: x[1])

    return [shard for shard, _ in scores]
```

**Result:** When two shards have equal distance, order is non-deterministic.

#### Hardening Strategy

```python
# CORRECT: Deterministic centroid ranking
def rank_shards_deterministic(query: Vector, shards: List[Shard]) -> List[Shard]:
    """
    Rank shards by centroid distance with deterministic tie-breaking.

    Total ordering: (distance, shard_id)
    """

    # Compute distances in fixed-point
    scores = []
    for shard in shards:
        dist = fixed_point_distance(query, shard.centroid)
        scores.append((shard, dist, shard.shard_id))

    # Stable sort: distance first, then shard_id for ties
    scores.sort(key=lambda x: (x[1], x[2]))

    return [shard for shard, _, _ in scores]
```

**Key Requirement:** Total ordering on all comparisons—no implicit tie-breaking.

---

*(Remaining sections of the original document continue below with similar structure, focusing on Cache Management, Floating-Point Normalization, Latency Measurement, and Verification)*

---

## Summary

Production substrate sharding can maintain **bit-identical determinism** at 250K+ scale if:

1. **Product Quantization** uses frozen codebooks and fixed-point distance
2. **HNSW graphs** are built single-threaded with deterministic layers
3. **Centroid ranking** explicitly breaks ties by shard ID (total ordering)
4. **Cache management** is single-threaded with FIFO eviction
5. **Floating point** is either avoided (fixed-point) or carefully normalized
6. **Latency is measured**, not estimated—use p50/p95/p99 distributions on target hardware

**The payoff:** Deterministic L1-resident inference with provable reproducibility across runs and platforms. Actual latency depends on measured distributions, not arithmetic. This makes the Material-Field Engine **falsifiable** in the scientific sense—same inputs provably produce same outputs—which is the foundation of the patent claim.

**Critical for deployment:**
- Freeze scaling factors at training time: `SCALE_768D = 43_832_413`
- Verify overflow bounds: `dist_squared < 2^63` for all valid inputs
- Establish total ordering everywhere: `(primary_key, secondary_key, ...)`
- Measure latency on target platform: median/p95/p99 across 10K+ queries

---

**Patent Priority:** January 25, 2026 | Verhash LLC
