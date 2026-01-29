#!/usr/bin/env python3
"""
Test Material-Field Engine with Reference Queries

Demonstrates system behavior with fixed reference queries:
- Simple factual prompts
- Common misconceptions
- Basic formula recall
- Date recall

Shows how elastic modulus mode affects exclusion behavior under a fixed substrate.
"""

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).parent))

from material_field_engine import (
    VerifiedSubstrate, Vector2D, MaterialFieldEngine
)


def test_science_fact(mode='multiplicative', sigma=0.5):
    """
    Query: 'What do plants need for photosynthesis?'
    Expected: Sunlight, water, carbon dioxide
    """
    print("\n" + "=" * 80)
    print("SCIENCE: What do plants need for photosynthesis?")
    print("=" * 80)
    print(f"Mode: {mode}, σ={sigma}")

    # Substrate: Verified science facts
    substrate = VerifiedSubstrate(
        elastic_modulus_mode=mode,
        elastic_modulus_sigma=sigma
    )

    # Verified answers (embeddings simulated as 2D for demo)
    substrate.add_verified_state(Vector2D(x=0.90, y=0.88, properties=None))  # "sunlight, water, CO2"

    # Candidates:
    # - Correct answer (close to substrate)
    # - Partially correct (medium distance)
    # - Common misconception (far from substrate)
    # - Creative but wrong (far, different direction)
    candidates = [
        (0.90, 0.88),  # Correct: sunlight, water, CO2
        (0.70, 0.65),  # Partial: mentions sunlight but incomplete
        (0.40, 0.35),  # Misconception: "plants eat soil"
        (0.15, 0.80),  # Creative wrong: "moonlight and air"
    ]

    engine = MaterialFieldEngine(
        substrate,
        lambda_min=0.35,
        lambda_max=1.00,
        inference_steps=8
    )

    engine.initialize_candidates(candidates)

    print("\nCandidate Knowledge States:")
    labels = [
        "Correct (sunlight+water+CO2)",
        "Partial (sunlight mentioned)",
        "Misconception (plants eat soil)",
        "Creative wrong (moonlight)"
    ]

    for i, (v, label) in enumerate(zip(engine.candidate_vectors, labels)):
        dist = v.distance_to(substrate.states[0])
        print(f"  {i}. {label:<30} | E={v.properties.elastic_modulus:.3f} | "
              f"σ_y={v.properties.yield_strength:.3f} | dist={dist:.3f}")

    results = engine.run_inference()

    print("\nResult:")
    if results['final_output']:
        winner_idx = next(i for i, c in enumerate(candidates)
                         if (c[0], c[1]) == (results['final_output'].x, results['final_output'].y))
        print(f"  Selected: {labels[winner_idx]}")
        print(f"  E={results['final_output'].properties.elastic_modulus:.3f}")
    else:
        print("  ABSTAINED (no candidate met grounding threshold)")

    print(f"  Excluded: {results['total_excluded']}")
    # Intentionally omit interpretive flags in stdout.


def test_geography_fact(mode='multiplicative', sigma=0.5):
    """
    Query: 'What is the capital of Texas?'
    Expected: Austin
    """
    print("\n" + "=" * 80)
    print("GEOGRAPHY: What is the capital of Texas?")
    print("=" * 80)
    print(f"Mode: {mode}, σ={sigma}")

    substrate = VerifiedSubstrate(
        elastic_modulus_mode=mode,
        elastic_modulus_sigma=sigma
    )

    # Verified answer
    substrate.add_verified_state(Vector2D(x=0.85, y=0.82, properties=None))  # "Austin"

    # Candidates:
    candidates = [
        (0.85, 0.82),  # Correct: Austin
        (0.75, 0.70),  # Common error: Houston (largest city)
        (0.40, 0.35),  # Wrong: Dallas
        (0.20, 0.15),  # Very wrong: Los Angeles (not even in Texas!)
    ]

    engine = MaterialFieldEngine(
        substrate,
        lambda_min=0.35,
        lambda_max=1.00,
        inference_steps=8
    )

    engine.initialize_candidates(candidates)

    print("\nCandidate Answers:")
    labels = [
        "Correct (Austin)",
        "Common error (Houston - largest city)",
        "Wrong (Dallas)",
        "Very wrong (Los Angeles)"
    ]

    for i, (v, label) in enumerate(zip(engine.candidate_vectors, labels)):
        dist = v.distance_to(substrate.states[0])
        print(f"  {i}. {label:<38} | E={v.properties.elastic_modulus:.3f} | "
              f"dist={dist:.3f}")

    results = engine.run_inference()

    print("\nResult:")
    if results['final_output']:
        winner_idx = next(i for i, c in enumerate(candidates)
                         if (c[0], c[1]) == (results['final_output'].x, results['final_output'].y))
        print(f"  Selected: {labels[winner_idx]}")
    else:
        print("  ABSTAINED")
    print(f"  Excluded: {results['total_excluded']}")


def test_math_concept(mode='multiplicative', sigma=0.5):
    """
    Query: 'What is the area formula for a rectangle?'
    Expected: length × width
    """
    print("\n" + "=" * 80)
    print("MATH: What is the area formula for a rectangle?")
    print("=" * 80)
    print(f"Mode: {mode}, σ={sigma}")

    substrate = VerifiedSubstrate(
        elastic_modulus_mode=mode,
        elastic_modulus_sigma=sigma
    )

    # Verified formula
    substrate.add_verified_state(Vector2D(x=0.92, y=0.90, properties=None))  # "length × width"

    candidates = [
        (0.92, 0.90),  # Correct: length × width
        (0.70, 0.68),  # Confused with perimeter: 2(l+w)
        (0.45, 0.42),  # Wrong: length + width
        (0.25, 0.20),  # Very wrong: confusing with volume
    ]

    engine = MaterialFieldEngine(
        substrate,
        lambda_min=0.40,
        lambda_max=1.10,
        inference_steps=8
    )

    engine.initialize_candidates(candidates)

    print("\nCandidate Formulas:")
    labels = [
        "Correct (length × width)",
        "Perimeter confusion (2(l+w))",
        "Wrong operation (length + width)",
        "Volume confusion (3D thinking)"
    ]

    for i, (v, label) in enumerate(zip(engine.candidate_vectors, labels)):
        print(f"  {i}. {label:<35} | E={v.properties.elastic_modulus:.3f}")

    results = engine.run_inference()

    print("\nResult:")
    if results['final_output']:
        winner_idx = next(i for i, c in enumerate(candidates)
                         if (c[0], c[1]) == (results['final_output'].x, results['final_output'].y))
        print(f"  Selected: {labels[winner_idx]}")
    else:
        print("  ABSTAINED")
    print(f"  Excluded: {results['total_excluded']}")


def test_historical_fact(mode='multiplicative', sigma=0.5):
    """
    Query: 'When did Christopher Columbus reach the Americas?'
    Expected: 1492
    """
    print("\n" + "=" * 80)
    print("HISTORY: When did Columbus reach the Americas?")
    print("=" * 80)
    print(f"Mode: {mode}, σ={sigma}")

    substrate = VerifiedSubstrate(
        elastic_modulus_mode=mode,
        elastic_modulus_sigma=sigma
    )

    # Verified date
    substrate.add_verified_state(Vector2D(x=0.88, y=0.86, properties=None))  # "1492"

    candidates = [
        (0.88, 0.86),  # Correct: 1492
        (0.75, 0.70),  # Close: 1490s range
        (0.50, 0.45),  # Common error: 1776 (confusing with US independence)
        (0.30, 0.25),  # Wrong century: 1500s
        (0.10, 0.12),  # Very wrong: 1942 (digit confusion)
    ]

    engine = MaterialFieldEngine(
        substrate,
        lambda_min=0.35,
        lambda_max=1.00,
        inference_steps=8
    )

    engine.initialize_candidates(candidates)

    print("\nCandidate Dates:")
    labels = [
        "Correct (1492)",
        "Approximate (1490s)",
        "Confusion (1776 - US independence)",
        "Wrong century (1500s)",
        "Digit swap (1942)"
    ]

    for i, (v, label) in enumerate(zip(engine.candidate_vectors, labels)):
        print(f"  {i}. {label:<38} | E={v.properties.elastic_modulus:.3f}")

    results = engine.run_inference()

    print("\nResult:")
    if results['final_output']:
        winner_idx = next(i for i, c in enumerate(candidates)
                         if (c[0], c[1]) == (results['final_output'].x, results['final_output'].y))
        print(f"  Selected: {labels[winner_idx]}")
    else:
        print("  ABSTAINED")
    print(f"  Excluded: {results['total_excluded']}")


def compare_modes():
    """Compare how different elastic modulus modes affect fixed reference queries"""
    print("\n" + "=" * 80)
    print("MODE COMPARISON: How does mode selection affect reference queries?")
    print("=" * 80)

    modes = [
        ('cosine', 0.5, "Cosine (direction only)"),
        ('multiplicative', 0.4, "Multiplicative σ=0.4 (tight - good for facts)"),
        ('multiplicative', 0.7, "Multiplicative σ=0.7 (loose - allows exploration)"),
    ]

    for mode, sigma, description in modes:
        print(f"\n{'─' * 80}")
        print(f"Testing with: {description}")
        print(f"{'─' * 80}")

        test_science_fact(mode, sigma)


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    Testing with Reference Queries                            ║
║                                                                              ║
║  Demonstrates material-field governance with simple example queries.        ║
║  Shows how elastic modulus mode affects factual recall vs. exploration.     ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

    # Default: use multiplicative mode with a fixed sigma for repeatability
    mode = 'multiplicative'
    sigma = 0.45

    if len(sys.argv) > 1:
        if sys.argv[1] == 'compare':
            compare_modes()
            sys.exit(0)
        elif sys.argv[1] == 'cosine':
            mode = 'cosine'
        elif sys.argv[1] == 'tight':
            sigma = 0.3
        elif sys.argv[1] == 'loose':
            sigma = 0.7

    print(f"\nUsing mode: {mode}, σ={sigma}")
    print("  (Run with 'compare' to see all modes side-by-side)\n")

    # Run all example tests
    test_science_fact(mode, sigma)
    test_geography_fact(mode, sigma)
    test_math_concept(mode, sigma)
    test_historical_fact(mode, sigma)
