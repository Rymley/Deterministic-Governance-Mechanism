#!/usr/bin/env python3
"""
Substrate Sharding Strategy for Material-Field Governance
Cache-Aware Design for 250,000+ Verified States

Goal: Keep most relevant substrate anchors in L1 cache (32-64KB)
while maintaining sub-microsecond access times and deterministic behavior.

Architecture:
  1. Hierarchical Sharding: Coarse → Fine-grained locality
  2. Semantic Clustering: Group related knowledge domains
  3. LRU Cache Management: Keep hot shards resident
  4. Deterministic Retrieval: Same query → same shards loaded

Reference Implementation - Verhash LLC
Patent Priority: January 25, 2026
"""

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Set
from collections import OrderedDict
import hashlib
import struct

from deterministic_rng import normal


# ==============================================================================
# CACHE ARCHITECTURE CONSTANTS
# ==============================================================================

L1_CACHE_SIZE = 32 * 1024      # 32 KB typical L1 data cache
L2_CACHE_SIZE = 256 * 1024     # 256 KB typical L2 cache
L3_CACHE_SIZE = 8 * 1024 * 1024 # 8 MB typical L3 cache

# Use explicit little-endian packing with fixed sizes for deterministic layout.
# Store x/y as float64 to match Python's `float` (C double on CPython).
VECTOR_SIZE_BYTES = 8 * 2       # 2 floats (x, y) × 8 bytes each = 16 bytes
                                # Real: 768 floats × 4 bytes = 3KB per vector

# L1 budget: Reserve space for working set + shard metadata
L1_BUDGET_VECTORS = (L1_CACHE_SIZE // 2) // VECTOR_SIZE_BYTES  # ~1000 vectors in L1
L2_BUDGET_VECTORS = (L2_CACHE_SIZE // 2) // VECTOR_SIZE_BYTES  # ~8000 vectors in L2


# ==============================================================================
# VECTOR REPRESENTATION
# ==============================================================================

@dataclass
class CompactVector:
    """
    Cache-optimized vector representation.

    In production:
    - Use float16 instead of float64 (halve memory)
    - Pack metadata into single 64-bit word
    - Align to cache line boundaries (64 bytes)
    """
    x: float
    y: float
    shard_id: int      # Which shard this belongs to
    domain_hash: int   # Semantic domain identifier

    def to_bytes(self) -> bytes:
        """Pack into contiguous bytes for cache efficiency"""
        return struct.pack('<ddIQ', float(self.x), float(self.y), int(self.shard_id), int(self.domain_hash))

    @staticmethod
    def from_bytes(data: bytes) -> 'CompactVector':
        """Unpack from contiguous bytes"""
        x, y, shard_id, domain_hash = struct.unpack('<ddIQ', data)
        return CompactVector(x, y, shard_id, domain_hash)

    def distance_to(self, other: 'CompactVector') -> float:
        """Euclidean distance"""
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def dot_product(self, other: 'CompactVector') -> float:
        """Normalized dot product (cosine similarity)"""
        self_norm = math.sqrt(self.x ** 2 + self.y ** 2)
        other_norm = math.sqrt(other.x ** 2 + other.y ** 2)

        if self_norm == 0 or other_norm == 0:
            return 0.0

        return (self.x * other.x + self.y * other.y) / (self_norm * other_norm)


# ==============================================================================
# SHARD STRUCTURE
# ==============================================================================

@dataclass
class SubstrateShard:
    """
    A shard is a contiguous block of verified states.

    Design:
    - Fixed size (e.g., 64 vectors = 1KB in 2D, 192KB in 768D)
    - Cache-line aligned
    - Sorted by locality (clustered semantically)
    - Immutable after creation (no reallocation)
    """
    shard_id: int
    domain: str                    # e.g., "biology", "physics", "geography"
    centroid: CompactVector        # Representative center of this shard
    vectors: List[CompactVector]   # The actual verified states
    size_bytes: int = 0            # Memory footprint

    def __post_init__(self):
        if self.size_bytes == 0:
            self.size_bytes = len(self.vectors) * VECTOR_SIZE_BYTES

    def is_l1_resident(self) -> bool:
        """Can this shard fit in L1?"""
        return self.size_bytes <= L1_CACHE_SIZE // 4  # Reserve 75% for other data

    def relevance_score(self, query_vector: CompactVector) -> float:
        """
        Compute relevance of this shard to query.
        Uses centroid distance as fast approximation.
        """
        return 1.0 / (1.0 + query_vector.distance_to(self.centroid))


# ==============================================================================
# HIERARCHICAL SHARDING STRATEGY
# ==============================================================================

class ShardedSubstrate:
    """
    Hierarchical substrate sharding with cache-aware retrieval.

    Architecture:
      Level 1 (L1 Cache): Active shard (~64 vectors)
      Level 2 (L2 Cache): Hot shards (~8 shards)
      Level 3 (L3 Cache): Warm shards (~100 shards)
      Level 4 (Main RAM): All shards (250K vectors / 64 = ~3906 shards)

    Retrieval Strategy:
      1. Hash query to domain (deterministic)
      2. Load relevant domain shards into L3
      3. Rank by centroid distance
      4. Prefetch top-k shards into L2
      5. Keep most relevant shard in L1
    """

    def __init__(self,
                 shard_size: int = 64,
                 l1_capacity: int = 1,      # Number of shards in L1
                 l2_capacity: int = 8,      # Number of shards in L2
                 l3_capacity: int = 100):    # Number of shards in L3

        self.shard_size = shard_size
        self.l1_capacity = l1_capacity
        self.l2_capacity = l2_capacity
        self.l3_capacity = l3_capacity

        # Storage hierarchy
        self.all_shards: Dict[int, SubstrateShard] = {}  # All shards (RAM)
        self.domain_index: Dict[str, List[int]] = {}     # Domain → shard IDs

        # Cache tiers (LRU with size limits)
        self.l1_cache: OrderedDict[int, SubstrateShard] = OrderedDict()
        self.l2_cache: OrderedDict[int, SubstrateShard] = OrderedDict()
        self.l3_cache: OrderedDict[int, SubstrateShard] = OrderedDict()

        # Statistics
        self.l1_hits = 0
        self.l2_hits = 0
        self.l3_hits = 0
        self.l1_misses = 0

    def add_shard(self, shard: SubstrateShard):
        """Add a shard to the substrate"""
        self.all_shards[shard.shard_id] = shard

        # Update domain index
        if shard.domain not in self.domain_index:
            self.domain_index[shard.domain] = []
        self.domain_index[shard.domain].append(shard.shard_id)

    def _get_domain_from_query(self, query_vector: CompactVector) -> str:
        """
        Deterministically map query to semantic domain.

        In production:
        - Use learned classifier
        - Or: hash query vector to domain buckets
        - Or: use query context/tags
        """
        # Simplified: use domain_hash embedded in query
        if query_vector.domain_hash % 3 == 0:
            return "biology"
        elif query_vector.domain_hash % 3 == 1:
            return "geography"
        else:
            return "physics"

    def _evict_lru(self, cache: OrderedDict, capacity: int):
        """Evict least recently used items to maintain capacity"""
        while len(cache) > capacity:
            cache.popitem(last=False)  # Remove oldest

    def _promote_to_l1(self, shard_id: int):
        """Move shard to L1 cache"""
        shard = self.all_shards[shard_id]

        # Remove from lower tiers
        self.l2_cache.pop(shard_id, None)
        self.l3_cache.pop(shard_id, None)

        # Add to L1 (most recent)
        self.l1_cache[shard_id] = shard
        self.l1_cache.move_to_end(shard_id)

        # Evict if over capacity
        self._evict_lru(self.l1_cache, self.l1_capacity)

    def _promote_to_l2(self, shard_id: int):
        """Move shard to L2 cache"""
        shard = self.all_shards[shard_id]

        # Remove from L3
        self.l3_cache.pop(shard_id, None)

        # Add to L2
        self.l2_cache[shard_id] = shard
        self.l2_cache.move_to_end(shard_id)

        self._evict_lru(self.l2_cache, self.l2_capacity)

    def _promote_to_l3(self, shard_id: int):
        """Move shard to L3 cache"""
        shard = self.all_shards[shard_id]

        # Add to L3
        self.l3_cache[shard_id] = shard
        self.l3_cache.move_to_end(shard_id)

        self._evict_lru(self.l3_cache, self.l3_capacity)

    def retrieve_relevant_shards(self,
                                  query_vector: CompactVector,
                                  top_k: int = 8) -> List[SubstrateShard]:
        """
        Retrieve most relevant shards for query.
        Cache-aware with deterministic retrieval.

        Strategy:
        1. Determine domain (deterministic hash)
        2. Get candidate shards from domain
        3. Rank by centroid distance
        4. Load top-k into L2
        5. Return in order of relevance
        """

        # Step 1: Get domain
        domain = self._get_domain_from_query(query_vector)

        # Step 2: Get candidate shard IDs from domain
        if domain not in self.domain_index:
            return []  # No shards for this domain

        candidate_ids = self.domain_index[domain]

        # Step 3: Rank shards by relevance
        shard_scores = [
            (shard_id, self.all_shards[shard_id].relevance_score(query_vector))
            for shard_id in candidate_ids
        ]
        shard_scores.sort(key=lambda x: x[1], reverse=True)

        # Step 4: Load top-k shards into cache hierarchy
        top_shard_ids = [shard_id for shard_id, _ in shard_scores[:top_k]]

        # Promote most relevant to L1
        if top_shard_ids:
            self._promote_to_l1(top_shard_ids[0])
            self.l1_hits += 1

        # Promote rest to L2
        for shard_id in top_shard_ids[1:]:
            if shard_id not in self.l2_cache:
                self._promote_to_l2(shard_id)
                self.l2_hits += 1

        # Return shards in order of relevance
        return [self.all_shards[sid] for sid in top_shard_ids]

    def get_l1_vectors(self) -> List[CompactVector]:
        """
        Get all vectors currently in L1 cache.
        This is the "hot path" for inference.
        """
        vectors = []
        for shard in self.l1_cache.values():
            vectors.extend(shard.vectors)
        return vectors

    def print_stats(self):
        """Print cache statistics"""
        print("\n" + "=" * 80)
        print("SUBSTRATE SHARDING STATISTICS")
        print("=" * 80)
        print(f"Total shards: {len(self.all_shards)}")
        print(f"Total vectors: {sum(len(s.vectors) for s in self.all_shards.values())}")
        print(f"Domains: {list(self.domain_index.keys())}")
        print()
        print(f"L1 Cache ({self.l1_capacity} shards max):")
        print(f"  Current: {len(self.l1_cache)} shards")
        print(f"  Vectors: {sum(len(s.vectors) for s in self.l1_cache.values())}")
        print(f"  Size: {sum(s.size_bytes for s in self.l1_cache.values())} bytes")
        print(f"  Hits: {self.l1_hits}")
        print()
        print(f"L2 Cache ({self.l2_capacity} shards max):")
        print(f"  Current: {len(self.l2_cache)} shards")
        print(f"  Vectors: {sum(len(s.vectors) for s in self.l2_cache.values())}")
        print(f"  Hits: {self.l2_hits}")
        print()
        print(f"L3 Cache ({self.l3_capacity} shards max):")
        print(f"  Current: {len(self.l3_cache)} shards")
        print(f"  Vectors: {sum(len(s.vectors) for s in self.l3_cache.values())}")
        print(f"  Hits: {self.l3_hits}")
        print("=" * 80)


# ==============================================================================
# DEMONSTRATION: 250K VERIFIED STATES
# ==============================================================================

def create_demo_substrate(num_vectors: int = 250000,
                          shard_size: int = 64) -> ShardedSubstrate:
    """
    Create a demo substrate with 250K verified states.
    Simulates three domains: biology, geography, physics.
    """

    print(f"Creating substrate with {num_vectors:,} verified states...")
    print(f"Shard size: {shard_size} vectors")
    print(f"Expected shards: {num_vectors // shard_size}")

    substrate = ShardedSubstrate(shard_size=shard_size)

    domains = ["biology", "geography", "physics"]
    vectors_per_domain = num_vectors // len(domains)

    shard_id = 0

    for domain_idx, domain in enumerate(domains):
        print(f"\nGenerating {vectors_per_domain:,} vectors for domain: {domain}")

        # Generate vectors clustered around domain centroid
        # Biology: centered around (0.8, 0.8)
        # Geography: centered around (0.5, 0.5)
        # Physics: centered around (0.2, 0.2)

        if domain == "biology":
            center_x, center_y = 0.8, 0.8
        elif domain == "geography":
            center_x, center_y = 0.5, 0.5
        else:
            center_x, center_y = 0.2, 0.2

        domain_hash = hashlib.sha256(domain.encode()).digest()[:8]
        domain_hash_int = int.from_bytes(domain_hash, 'big')
        domain_seed = b"demo_substrate|" + domain.encode("utf-8")

        # Create shards for this domain
        for i in range(0, vectors_per_domain, shard_size):
            vectors = []

            for j in range(shard_size):
                if i + j >= vectors_per_domain:
                    break

                # Generate vector with small random offset
                global_idx = i + j
                x = center_x + normal(domain_seed, global_idx * 4, mean=0.0, std=0.1)
                y = center_y + normal(domain_seed, global_idx * 4 + 2, mean=0.0, std=0.1)

                vec = CompactVector(
                    x=x,
                    y=y,
                    shard_id=shard_id,
                    domain_hash=domain_hash_int
                )
                vectors.append(vec)

            # Compute centroid
            if vectors:
                centroid_x = sum(v.x for v in vectors) / len(vectors)
                centroid_y = sum(v.y for v in vectors) / len(vectors)
                centroid = CompactVector(centroid_x, centroid_y, shard_id, domain_hash_int)

                shard = SubstrateShard(
                    shard_id=shard_id,
                    domain=domain,
                    centroid=centroid,
                    vectors=vectors
                )

                substrate.add_shard(shard)
                shard_id += 1

    print(f"\nSubstrate created: {len(substrate.all_shards)} shards")
    return substrate


def demo_query_performance():
    """
    Demonstrate query performance with sharded substrate.
    """

    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║               Substrate Sharding Demonstration                               ║
║                                                                              ║
║   Exercises deterministic sharding and cache residency over                  ║
║   a large verified substrate. Illustrates how relevance                      ║
║   ranking and promotion constrain active context.                            ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

    # Create substrate
    substrate = create_demo_substrate(num_vectors=250000, shard_size=64)

    substrate.print_stats()

    # Simulate queries from different domains
    print("\n" + "=" * 80)
    print("QUERY SIMULATION")
    print("=" * 80)

    # Biology query
    print("\nQuery 1: Biology domain (photosynthesis)")
    bio_query = CompactVector(x=0.82, y=0.78, shard_id=-1, domain_hash=0)
    relevant_shards = substrate.retrieve_relevant_shards(bio_query, top_k=8)

    print(f"  Retrieved {len(relevant_shards)} relevant shards")
    print(f"  L1 vectors available: {len(substrate.get_l1_vectors())}")
    print(f"  Most relevant shard: {relevant_shards[0].shard_id} (domain: {relevant_shards[0].domain})")

    # Geography query
    print("\nQuery 2: Geography domain (capitals)")
    geo_query = CompactVector(x=0.48, y=0.52, shard_id=-1, domain_hash=1)
    relevant_shards = substrate.retrieve_relevant_shards(geo_query, top_k=8)

    print(f"  Retrieved {len(relevant_shards)} relevant shards")
    print(f"  L1 vectors available: {len(substrate.get_l1_vectors())}")
    print(f"  Most relevant shard: {relevant_shards[0].shard_id} (domain: {relevant_shards[0].domain})")

    # Physics query
    print("\nQuery 3: Physics domain (mechanics)")
    phys_query = CompactVector(x=0.18, y=0.22, shard_id=-1, domain_hash=2)
    relevant_shards = substrate.retrieve_relevant_shards(phys_query, top_k=8)

    print(f"  Retrieved {len(relevant_shards)} relevant shards")
    print(f"  L1 vectors available: {len(substrate.get_l1_vectors())}")
    print(f"  Most relevant shard: {relevant_shards[0].shard_id} (domain: {relevant_shards[0].domain})")

    # Print final stats
    substrate.print_stats()

    # Memory footprint analysis
    print("\n" + "=" * 80)
    print("MEMORY FOOTPRINT ANALYSIS")
    print("=" * 80)

    l1_size = sum(s.size_bytes for s in substrate.l1_cache.values())
    l2_size = sum(s.size_bytes for s in substrate.l2_cache.values())
    total_size = sum(s.size_bytes for s in substrate.all_shards.values())

    print(f"L1 Cache: {l1_size:,} bytes ({l1_size/1024:.2f} KB)")
    print(f"  Fits in L1? {l1_size <= L1_CACHE_SIZE}")
    print(f"  L1 capacity: {L1_CACHE_SIZE:,} bytes ({L1_CACHE_SIZE/1024:.2f} KB)")
    print()
    print(f"L2 Cache: {l2_size:,} bytes ({l2_size/1024:.2f} KB)")
    print(f"  Fits in L2? {l2_size <= L2_CACHE_SIZE}")
    print(f"  L2 capacity: {L2_CACHE_SIZE:,} bytes ({L2_CACHE_SIZE/1024:.2f} KB)")
    print()
    print(f"Total Substrate: {total_size:,} bytes ({total_size/1024/1024:.2f} MB)")
    print()
    print(f"Cache efficiency: {(l1_size + l2_size) / total_size * 100:.2f}% in fast cache")

    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    print("""
1. CACHE RESIDENCY:
   - L1 holds 1 shard (64 vectors) = most relevant for current query
   - L2 holds 8 shards (512 vectors) = domain-relevant context
   - Working set: 576 vectors in fast cache (< 10 KB in 2D)

2. RETRIEVAL STRATEGY:
   - Deterministic domain mapping (hash-based)
   - Centroid-based relevance ranking
   - LRU eviction maintains hot shards

3. SCALING TO 768D:
   - 768 floats × 4 bytes = 3 KB per vector
   - L1: 64 vectors × 3 KB = 192 KB (fits in L2, not L1)
   - Strategy: Reduce L1 capacity to 10-16 vectors for true L1 residence

4. PRODUCTION OPTIMIZATIONS:
   - Use float16 (half precision) → 1.5 KB per vector
   - Quantize to int8 → 768 bytes per vector
   - Product quantization → 64-128 bytes per vector
   - With PQ: L1 can hold 256-512 vectors (16-32 KB)

5. INFERENCE PATH:
   - Query arrives → Hash to domain → Load L2 shards
   - Rank by centroid → Promote top shard to L1
   - Inference runs on L1-resident vectors only
   - Sub-microsecond retrieval, deterministic, cache-friendly
""")


if __name__ == "__main__":
    demo_query_performance()
