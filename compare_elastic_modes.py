#!/usr/bin/env python3
"""
Comparison Demo: Direction-Only vs. Composite Elastic Modulus

This script demonstrates the behavioral differences between different
elastic modulus computation modes:
1. Cosine: Direction-only scalar (distance does not reduce E)
2. Multiplicative: Composite scalar (alignment × proximity)
3. RBF: Proximity-only scalar (direction ignored)

Shows how kernel choice changes exclusion behavior under the same candidates.
"""

import sys
from pathlib import Path

# Avoid Windows console UnicodeEncodeError for optional fancy output.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from material_field_engine import (
    VerifiedSubstrate, Vector2D, MaterialFieldEngine, PhaseTransitionController
)


def print_section_header(title):
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def print_candidate_properties(vectors, title="Candidate Properties"):
    print(f"\n{title}:")
    print(f"{'Vector':<20} | {'Distance':<10} | {'E':<8} | {'σ_y':<8} | {'ε':<8}")
    print("-" * 70)
    for i, v in enumerate(vectors):
        # Compute distance from substrate (assume substrate is at origin for display)
        dist = (v.x**2 + v.y**2)**0.5
        print(f"({v.x:.3f}, {v.y:.3f}){'':>6} | "
              f"{dist:<10.4f} | "
              f"{v.properties.elastic_modulus:<8.4f} | "
              f"{v.properties.yield_strength:<8.4f} | "
              f"{v.properties.strain:<8.4f}")


def run_comparison_scenario(scenario_name, substrate_positions, candidate_positions,
                            lambda_min=0.30, lambda_max=0.90, total_steps=8):
    """
    Run the same scenario with different elastic modulus modes and compare results.
    """
    print_section_header(f"SCENARIO: {scenario_name}")

    modes = [
        ('cosine', None, "Pure Angular Alignment"),
        ('multiplicative', 0.3, "Multiplicative (σ=0.3, tight binding)"),
        ('multiplicative', 0.5, "Multiplicative (σ=0.5, balanced)"),
        ('multiplicative', 0.8, "Multiplicative (σ=0.8, loose binding)"),
        ('rbf', 0.5, "Pure Proximity (RBF, σ=0.5)")
    ]

    results_summary = []

    for mode, sigma, description in modes:
        print(f"\n{'-' * 80}")
        print(f"MODE: {description}")
        print(f"{'-' * 80}")

        # Create substrate with this mode
        if sigma is None:
            substrate = VerifiedSubstrate(elastic_modulus_mode=mode)
        else:
            substrate = VerifiedSubstrate(elastic_modulus_mode=mode,
                                         elastic_modulus_sigma=sigma)

        # Add substrate states
        for pos in substrate_positions:
            substrate.add_verified_state(Vector2D(x=pos[0], y=pos[1], properties=None))

        # Create engine
        engine = MaterialFieldEngine(substrate, lambda_min, lambda_max, total_steps)

        # Initialize candidates
        engine.initialize_candidates(candidate_positions)

        # Print initial properties
        print_candidate_properties(engine.candidate_vectors, "Initial Properties")

        # Run inference
        results = engine.run_inference()

        # Print results
        print(f"\nResults:")
        print(f"  Final Output: ", end="")
        if results['final_output']:
            print(f"({results['final_output'].x:.3f}, {results['final_output'].y:.3f})")
            print(f"  Final E: {results['final_output'].properties.elastic_modulus:.4f}")
        else:
            print("ABSTAINED (no surviving candidates)")
        print(f"  Total Excluded: {results['total_excluded']}")

        # Store summary
        results_summary.append({
            'mode': description,
            'output': results['final_output'],
            'excluded': results['total_excluded'],
            'abstained': results['abstained']
        })

    # Print comparison summary
    print_section_header("COMPARISON SUMMARY")
    print(f"\n{'Mode':<45} | {'Output':<20} | {'Excluded':<10}")
    print("-" * 80)
    for summary in results_summary:
        if summary['output']:
            output_str = f"({summary['output'].x:.3f}, {summary['output'].y:.3f})"
        else:
            output_str = "ABSTAINED"
        print(f"{summary['mode']:<45} | {output_str:<20} | {summary['excluded']:<10}")


def scenario_1_aligned_near_far():
    """
    Scenario 1: Multiple aligned candidates at different distances

    Tests whether distance affects selection when all candidates point in same direction.
    """
    substrate_positions = [
        (0.95, 0.92)  # Single verified state
    ]

    candidate_positions = [
        (0.95, 0.92),  # Exact match - distance 0
        (0.85, 0.82),  # Very close, aligned - distance ≈0.14
        (0.50, 0.50),  # Medium distance, aligned - distance ≈0.62
        (0.35, 0.30),  # Far, aligned - distance ≈0.86
        (0.10, 0.10),  # Very far, aligned - distance ≈1.18
    ]

    run_comparison_scenario(
        "Aligned Candidates at Different Distances",
        substrate_positions,
        candidate_positions
    )


def scenario_2_aligned_vs_close():
    """
    Scenario 2: Aligned-but-far vs. Close-but-misaligned

    Tests the trade-off between angular alignment and proximity.
    """
    substrate_positions = [
        (1.0, 0.0)  # Substrate on positive X-axis
    ]

    candidate_positions = [
        (0.95, 0.0),   # Very close and aligned (distance ≈0.05)
        (0.80, 0.40),  # Close but misaligned (distance ≈0.29, angle ≈26°)
        (0.50, 0.50),  # Medium distance, 45° angle (distance ≈0.71)
        (0.20, 0.80),  # Far and misaligned (distance ≈1.13, angle ≈76°)
    ]

    run_comparison_scenario(
        "Aligned vs. Close Trade-off",
        substrate_positions,
        candidate_positions
    )


def scenario_3_multiple_substrate_states():
    """
    Scenario 3: Multiple substrate states (cluster)

    Tests behavior when substrate forms a semantic cluster.
    """
    substrate_positions = [
        (0.90, 0.85),
        (0.88, 0.92),
        (0.95, 0.88)
    ]

    candidate_positions = [
        (0.91, 0.88),  # Center of cluster
        (0.70, 0.70),  # Medium distance
        (0.50, 0.50),  # Far from cluster
        (0.10, 0.10),  # Very far
    ]

    run_comparison_scenario(
        "Multiple Substrate States (Cluster)",
        substrate_positions,
        candidate_positions
    )


def scenario_4_noise_filtering():
    """
    Scenario 4: Real signal vs. Noise

    One grounded candidate vs. multiple noise candidates.
    """
    substrate_positions = [
        (0.88, 0.85)  # Verified obstacle pattern
    ]

    candidate_positions = [
        (0.88, 0.83),  # Real obstacle (very close match)
        (0.15, 0.12),  # Sensor noise (far, low confidence)
        (0.22, 0.18),  # More sensor noise
        (0.08, 0.25),  # More sensor noise
    ]

    run_comparison_scenario(
        "Real Signal vs. Noise Filtering",
        substrate_positions,
        candidate_positions,
        lambda_min=0.25,
        lambda_max=0.95,
        total_steps=8
    )


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║         ELASTIC MODULUS MODE COMPARISON: Cosine vs. Distance-Aware          ║
║                                                                              ║
║  This demo shows how different E computation modes affect which candidates  ║
║  survive phase transitions and become final outputs.                        ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

    # Run all scenarios
    scenario_1_aligned_near_far()
    print("\n\n")

    scenario_2_aligned_vs_close()
    print("\n\n")

    scenario_3_multiple_substrate_states()
    print("\n\n")

    scenario_4_noise_filtering()

    # Final guidance
    print_section_header("INTERPRETATION GUIDE")
    print("""
Key Observations:

1. COSINE MODE (Direction-Only):
   - E is a function of alignment only.
   - Distance/magnitude does not reduce E.
   - Exclusion pressure is applied against σ_y, with strain/stress carrying the distance signal.

2. MULTIPLICATIVE MODE (Composite / Gated):
   - E is computed as (alignment_term × proximity_term).
   - This composes alignment and distance into a single scalar gate.
   - σ is the proximity bandwidth; smaller σ contracts the effective binding radius.

3. RBF MODE (Proximity-Only):
   - E is a function of distance only.
   - Direction does not contribute to E.

Example parameterizations (explicit, recorded in the run config):
   - Multiplicative σ=0.30
   - Multiplicative σ=0.50
   - Multiplicative σ=0.80
""")

    print("\n" + "=" * 80)
    print("Configuration: Edit config.json 'elastic_modulus' section to set mode/sigma")
    print("=" * 80)
