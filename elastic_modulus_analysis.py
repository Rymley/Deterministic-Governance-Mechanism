"""
ANALYSIS: Elastic Modulus Saturation and Distance-Aware Alternatives

Current Issue:
--------------
Cosine similarity (normalized dot product) creates directional alignment measure
but ignores magnitude/distance. In low-dimensional spaces, this causes:

1. E ≈ 1.0 for almost all nonzero vectors (direction dominates)
2. Semantic class selection stuck in high-yield band (0.90-0.98)
3. Exclusion driven entirely by strain (ε), not true substrate support
4. System behaves as "distance filter wearing alignment mask"

Example from test output:
  Vector (0.35, 0.30): E=0.9991, ε=0.8628
  Vector (0.50, 0.50): E=0.9999, ε=0.6155
  Vector (0.10, 0.10): E=0.9999, ε=1.1811

All have near-unit elastic modulus despite vastly different distances!

Why This Happens:
-----------------
Current formula: E = (cos_similarity + 1.0) / 2.0

For any vector roughly pointing toward substrate direction:
  cos(θ) ≈ 1.0 when θ ≈ 0° (regardless of distance)
  → E ≈ 1.0

This is geometrically correct for angular alignment, but semantically
incomplete: a vector 10 units away in the same direction should NOT
have the same "rigidity" as one 0.01 units away.

Proposed Alternative: Distance-Aware Elastic Modulus
----------------------------------------------------

Blend proximity and alignment using different kernel functions:

Option 1: Multiplicative Coupling
----------------------------------
E = alignment_term × proximity_term

Where:
  alignment_term = (cos_similarity + 1.0) / 2.0  # Angular alignment
  proximity_term = exp(-distance² / 2σ²)          # Gaussian proximity kernel

This creates a proper "substrate field" where:
- High E requires BOTH correct direction AND close distance
- Far vectors (even aligned) get low E → low yield strength → early fracture
- Close vectors (even slightly misaligned) get high E → survive longer

Example with σ = 0.5:

Vector          | cos_sim | distance | alignment | proximity | E (product)
----------------|---------|----------|-----------|-----------|------------
(0.95, 0.92)    | 1.000   | 0.000    | 1.000     | 1.000     | 1.000
(0.35, 0.30)    | 0.999   | 0.863    | 0.999     | 0.037     | 0.037
(0.50, 0.50)    | 0.999   | 0.615    | 0.999     | 0.286     | 0.286
(0.10, 0.10)    | 0.999   | 1.181    | 0.999     | 0.002     | 0.002

Now E genuinely reflects "grounding strength" rather than just angle!

Option 2: Harmonic Mean (Balanced)
-----------------------------------
E = 2 / (1/alignment + 1/proximity)

Properties:
- Both factors must be high for high E
- More balanced than multiplication
- Penalizes imbalance (one factor low → E low)

Option 3: Weighted Sum (Tunable)
---------------------------------
E = α·alignment + (1-α)·proximity

Where α controls the balance:
  α = 0.7 → favor alignment (original behavior)
  α = 0.5 → equal weight
  α = 0.3 → favor proximity (tight substrate binding)

Option 4: RBF Kernel (Pure Proximity)
--------------------------------------
E = exp(-distance² / 2σ²)

Extreme case: ignore direction entirely, only distance matters.
Useful when substrate represents "allowed regions" rather than directions.

Implementation Example:
-----------------------
"""

import sys

# Avoid Windows console UnicodeEncodeError for optional fancy output.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import math

def compute_elastic_modulus_distance_aware(
    candidate_vector, 
    substrate_vectors,
    mode='multiplicative',
    sigma=0.5,
    alpha=0.5
):
    """
    Compute elastic modulus with distance awareness.
    
    Args:
        candidate_vector: (x, y) coordinates
        substrate_vectors: List of verified substrate (x, y) coordinates
        mode: 'multiplicative', 'harmonic', 'weighted', 'rbf'
        sigma: Gaussian kernel bandwidth (for proximity term)
        alpha: Weight for alignment in 'weighted' mode
    
    Returns:
        E: Elastic modulus in [0, 1]
    """
    if not substrate_vectors:
        return 0.5  # Default
    
    # Find nearest substrate vector
    best_alignment = -1.0
    best_distance = float('inf')
    
    for substrate in substrate_vectors:
        # Cosine similarity (alignment)
        dot_prod = candidate_vector[0] * substrate[0] + candidate_vector[1] * substrate[1]
        cand_norm = math.sqrt(candidate_vector[0] ** 2 + candidate_vector[1] ** 2)
        subs_norm = math.sqrt(substrate[0] ** 2 + substrate[1] ** 2)
        
        if cand_norm > 0 and subs_norm > 0:
            cos_sim = dot_prod / (cand_norm * subs_norm)
        else:
            cos_sim = 0.0
        
        # Euclidean distance
        dist = math.sqrt(
            (candidate_vector[0] - substrate[0]) ** 2
            + (candidate_vector[1] - substrate[1]) ** 2
        )
        
        # Track best (highest alignment, lowest distance)
        if cos_sim > best_alignment:
            best_alignment = cos_sim
            best_distance = dist
    
    # Compute terms
    alignment_term = (best_alignment + 1.0) / 2.0  # Map [-1,1] → [0,1]
    proximity_term = math.exp(-(best_distance ** 2) / (2 * (sigma ** 2)))
    
    # Combine based on mode
    if mode == 'multiplicative':
        E = alignment_term * proximity_term
    
    elif mode == 'harmonic':
        if alignment_term > 0 and proximity_term > 0:
            E = 2.0 / (1.0/alignment_term + 1.0/proximity_term)
        else:
            E = 0.0
    
    elif mode == 'weighted':
        E = alpha * alignment_term + (1 - alpha) * proximity_term
    
    elif mode == 'rbf':
        E = proximity_term  # Pure proximity
    
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    return min(1.0, max(0.0, E))


# Demonstration
print("=" * 80)
print("ELASTIC MODULUS: COSINE-ONLY vs. DISTANCE-AWARE")
print("=" * 80)

substrate = [(0.95, 0.92)]
test_vectors = [
    (0.95, 0.92, "Substrate (exact match)"),
    (0.94, 0.91, "Very close, aligned"),
    (0.50, 0.50, "Medium distance, aligned"),
    (0.35, 0.30, "Far, aligned"),
    (0.10, 0.10, "Very far, aligned"),
]

print("\nSubstrate: (0.95, 0.92)")
print("\n" + "-" * 80)
print(f"{'Vector':<20} | {'Distance':<10} | {'E (cosine)':<12} | {'E (mult.)':<12} | {'E (harm.)':<12}")
print("-" * 80)

for x, y, label in test_vectors:
    # Cosine-only (current implementation)
    dot = x * substrate[0][0] + y * substrate[0][1]
    norm_v = math.sqrt(x ** 2 + y ** 2)
    norm_s = math.sqrt(substrate[0][0] ** 2 + substrate[0][1] ** 2)
    cos_sim = dot / (norm_v * norm_s) if norm_v > 0 and norm_s > 0 else 0
    E_cosine = (cos_sim + 1.0) / 2.0
    
    # Distance
    distance = math.sqrt((x - substrate[0][0]) ** 2 + (y - substrate[0][1]) ** 2)
    
    # Distance-aware modes
    E_mult = compute_elastic_modulus_distance_aware(
        (x, y), substrate, mode='multiplicative', sigma=0.5
    )
    E_harm = compute_elastic_modulus_distance_aware(
        (x, y), substrate, mode='harmonic', sigma=0.5
    )
    
    print(f"{label:<20} | {distance:>8.4f}   | {E_cosine:>10.4f}   | {E_mult:>10.4f}   | {E_harm:>10.4f}")

print("\n" + "=" * 80)
print("ANALYSIS:")
print("=" * 80)
print("• Cosine-only: E ≈ 1.0 for all (saturated)")
print("• Multiplicative: E reflects true grounding (distance × alignment)")
print("• Harmonic: Balanced, requires both factors high")
print("\nConclusion: Distance-aware E creates meaningful semantic classes,")
print("            enabling the yield strength mechanism to work as intended.")
print("=" * 80)

"""
Configuration Recommendation:
-----------------------------

For production deployment, use multiplicative coupling with tunable σ:

config.json addition:
{
  "elastic_modulus": {
    "mode": "multiplicative",
    "sigma": 0.5,
    "comment": "Gaussian kernel bandwidth for proximity term"
  }
}

Tuning guidelines:
- Small σ (0.3): Tight substrate binding, only very close vectors get high E
- Medium σ (0.5): Balanced, moderate proximity required
- Large σ (1.0): Loose binding, distance matters less

Different σ for different semantic classes:
- Verified facts: σ = 0.3 (tight binding)
- Contextual: σ = 0.5 (moderate)
- Creative: σ = 0.8 (loose, allow exploration)

This creates a "substrate field strength" model where different regions
of semantic space have different binding characteristics.
"""
