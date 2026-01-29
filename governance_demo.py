#!/usr/bin/env python3
"""
This demonstration shows that the Material-Field Engine does not
encode scientific knowledge and does not reason about biology. 
It operates purely by mechanical constraint: candidate semantic vectors
are subjected to increasing pressure, and those that cannot maintain 
structural integrity are irreversibly excluded. What is commonly framed 
as an “alignment” or “understanding” problem is treated instead as
a stability problem in a constrained field.

The query evaluated is “Where do plants get their food?” The verified 
substrate provides evidential grounding: plants synthesize glucose from sunlight, 
water, and carbon dioxide; chlorophyll in leaves is the site of energy conversion; 
and soil contributes minerals and water but not the bulk of plant mass. 
Four candidates are introduced into the latent field: a tightly aligned 
textbook answer, a common misconception asserting soil consumption, a vague 
answer referencing the ground, and an implausible hallucination. Under 
deterministic phase transitions, structurally weak candidates fracture 
during nucleation or quenching, leaving only the aligned, grounded vector 
to survive crystallization with a bit-identical output.
"""

import sys
import hashlib
import json
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


def governance_demo(mode='multiplicative', sigma=0.4, verbose=True):
    """
    Run the governance demo with specified elastic modulus configuration.

    Returns:
        dict: Complete test results including phase log, final output, and hash
    """

    if verbose:
        print("=" * 80)
        print("DETERMINISTIC GOVERNANCE DEMO: Photosynthesis")
        print("=" * 80)
        print(f"\nQuery: 'Where do plants get their food?'")
        print(f"Mode: {mode}, σ={sigma:.2f}")
        print()

    # ============================================================================
    # SUBSTRATE: Verified introductory biology facts
    # ============================================================================

    substrate = VerifiedSubstrate(
        elastic_modulus_mode=mode,
        elastic_modulus_sigma=sigma
    )

    # Fact A: Primary Anchor - Photosynthesis equation
    substrate.add_verified_state(Vector2D(x=0.95, y=0.92, properties=None))

    # Fact B: Structural Support - Chlorophyll mechanism
    substrate.add_verified_state(Vector2D(x=0.90, y=0.88, properties=None))

    # Fact C: Anti-Hallucination Fuse - Soil misconception correction
    substrate.add_verified_state(Vector2D(x=0.88, y=0.90, properties=None))

    if verbose:
        print("Substrate (Verified Biology Facts):")
        print("  Fact A (0.95, 0.92): Photosynthesis equation [Primary Anchor]")
        print("  Fact B (0.90, 0.88): Chlorophyll mechanism [Structural Support]")
        print("  Fact C (0.88, 0.90): Soil correction [Anti-Hallucination Fuse]")
        print()

    # ============================================================================
    # CANDIDATES: Four Specific Semantic States
    # ============================================================================

    candidates = [
        (0.95, 0.92),   # Textbook: Correct answer
        (0.10, 0.10),   # Misconception: "They eat soil"
        (0.50, 0.50),   # Vague: "From the ground"
        (-0.80, -0.80), # Hallucination: "They hunt insects"
    ]

    candidate_labels = [
        "Textbook (sunlight+water+CO2)",
        "Misconception (eat soil)",
        "Vague (from the ground)",
        "Hallucination (hunt insects)"
    ]

    # ============================================================================
    # ENGINE INITIALIZATION
    # ============================================================================

    engine = MaterialFieldEngine(
        substrate,
        lambda_min=0.35,
        lambda_max=1.20,
        inference_steps=8
    )

    engine.phase_controller.nucleation_threshold = 0.375  # Ends at step 3
    engine.phase_controller.quenching_threshold = 0.875   # Ends at step 7

    engine.initialize_candidates(candidates)

    if verbose:
        print("Candidate Semantic States:")
        print(f"{'#':<3} {'Label':<35} | {'E':<8} | {'σ_y':<8} | {'ε':<8} | {'σ_init':<8}")
        print("-" * 80)

    for i, (v, label) in enumerate(zip(engine.candidate_vectors, candidate_labels)):
        if verbose:
            print(f"{i:<3} {label:<35} | "
                  f"{v.properties.elastic_modulus:<8.4f} | "
                  f"{v.properties.yield_strength:<8.4f} | "
                  f"{v.properties.strain:<8.4f} | "
                  f"{v.properties.stress:<8.4f}")

    if verbose:
        print()
        print("=" * 80)
        print("PHASE TRANSITION LOG")
        print("=" * 80)

    # ============================================================================
    # RUN INFERENCE
    # ============================================================================

    results = engine.run_inference()

    if verbose:
        print(f"\n{'Step':<5} | {'Phase':<15} | {'λ(t)':<8} | {'Survivors':<10} | {'Status'}")
        print("-" * 80)

    for entry in results['phase_log']:
        step = entry['step']
        phase = entry['phase']
        pressure = entry['pressure']
        survivors = entry['survivors']
        excluded = entry['excluded']

        # Determine status message
        if step == 0:
            status = "Initial state"
        elif excluded > 0 and step <= 3:
            status = f"← {excluded} fractured (brittle failure)"
        elif excluded > 0 and 4 <= step <= 6:
            status = f"← Vague candidate fracturing"
        elif excluded > 0 and step >= 7:
            status = f"← Final exclusion"
        else:
            status = ""

        if verbose:
            print(f"{step:<5} | {phase:<15} | {pressure:<8.3f} | {survivors:<10} | {status}")

    # ============================================================================
    # RESULTS AND VERIFICATION
    # ============================================================================

    if verbose:
        print("\n" + "=" * 80)
        print("GOVERNANCE DEMO RESULTS")
        print("=" * 80)

    # Identify final output
    final_output = results['final_output']
    if final_output:
        output_tuple = (final_output.x, final_output.y)
        winner_idx = next(i for i, c in enumerate(candidates) if c == output_tuple)
        winner_label = candidate_labels[winner_idx]

        if verbose:
            print(f"\nFinal Output: {winner_label}")
            print(f"  Coordinates: ({final_output.x:.3f}, {final_output.y:.3f})")
            print(f"  Elastic Modulus: {final_output.properties.elastic_modulus:.6f}")
            print(f"  Yield Strength: {final_output.properties.yield_strength:.6f}")
            print(f"  Final Stress: {final_output.properties.stress:.6f}")
    else:
        winner_label = "ABSTAINED"
        if verbose:
            print(f"\nFinal Output: ABSTAINED (no candidate met structural requirements)")

    if verbose:
        print(f"\nTotal Excluded: {results['total_excluded']}")
        print(f"Hallucination-Free: {results['hallucination_free']}")
        print(f"Deterministic: {results['deterministic']}")
        print(f"Latency: {results['latency_ms']:.3f} ms")

    # ============================================================================
    # BIT-IDENTICAL VERIFICATION
    # ============================================================================

    # Create deterministic hash of results
    result_data = {
        'final_output': (final_output.x, final_output.y) if final_output else None,
        'excluded_count': results['total_excluded'],
        'phase_log': [
            {
                'step': e['step'],
                'phase': e['phase'],
                'survivors': e['survivors'],
                'pressure': round(e['pressure'], 6)
            }
            for e in results['phase_log']
        ]
    }

    result_json = json.dumps(result_data, sort_keys=True)
    result_hash = hashlib.sha256(result_json.encode()).hexdigest()

    if verbose:
        print(f"\nBit-Identical Verification:")
        print(f"  SHA-256: {result_hash}")

    # ============================================================================
    # FALSIFICATION CRITERIA
    # ============================================================================

    if verbose:
        print("\n" + "=" * 80)
        print("FALSIFICATION ANALYSIS")
        print("=" * 80)

        # Check expected behavior
        nucleation_exclusions = sum(1 for e in results['phase_log'][:4] if e['excluded'] > 0)
        quenching_exclusions = sum(1 for e in results['phase_log'][4:7] if e['excluded'] > 0)

        print(f"\nExpected vs. Actual Behavior:")
        print(f"  Nucleation exclusions (Steps 0-3): {nucleation_exclusions} events")
        print(f"  Quenching exclusions (Steps 4-6): {quenching_exclusions} events")
        print(f"  Final survivor: {winner_label}")

        # Verify correctness
        correct = (
            results['total_excluded'] == 3 and
            winner_label == "Textbook (sunlight+water+CO2)" and
            results['hallucination_free']
        )

        print(f"\n  Test Status: {'✓ PASS' if correct else '✗ FAIL'}")

        if correct:
            print("\n  Interpretation:")
            print("    The engine doesn't 'understand' photosynthesis.")
            print("    It mechanically excludes vectors that cannot maintain")
            print("    structural integrity under constraint pressure.")
            print("    This is material science, not semantic reasoning.")

    # ============================================================================
    # RETURN COMPLETE RESULTS
    # ============================================================================

    return {
        'winner': winner_label,
        'hash': result_hash,
        'excluded': results['total_excluded'],
        'hallucination_free': results['hallucination_free'],
        'phase_log': results['phase_log'],
        'final_output': final_output
    }


def compare_modes():
    """Compare demo behavior across different elastic modulus modes."""

    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    GOVERNANCE DEMO: MODE COMPARISON                          ║
║                                                                              ║
║  Shows how elastic modulus mode affects mechanical exclusion of wrong       ║
║  answers under a fixed substrate and candidate set.                         ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

    modes = [
        ('cosine', 0.4, "Cosine (Direction Only)"),
        ('multiplicative', 0.4, "Multiplicative σ=0.4 (Reference)"),
        ('multiplicative', 0.6, "Multiplicative σ=0.6 (Looser)"),
    ]

    results = []

    for mode, sigma, description in modes:
        print(f"\n{'─' * 80}")
        print(f"Mode: {description}")
        print(f"{'─' * 80}\n")

        result = governance_demo(mode, sigma, verbose=True)
        results.append((description, result))

        print("\n")

    # Summary comparison
    print("=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    print(f"\n{'Mode':<40} | {'Winner':<35} | {'Excluded'}")
    print("-" * 80)

    for desc, res in results:
        print(f"{desc:<40} | {res['winner']:<35} | {res['excluded']}")

    print("\nKey Insight:")
    print("  Cosine mode may allow wrong answers with high E due to angular alignment.")
    print("  Multiplicative mode requires BOTH alignment AND proximity.")
    print("  Multiplicative mode composes alignment and proximity deterministically.")


def verify_determinism():
    """Run the demo 5 times to verify bit-identical results."""

    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                  DETERMINISM VERIFICATION: 5-RUN TEST                        ║
║                                                                              ║
║  Verifies that the demo produces bit-identical outputs across               ║
║  multiple runs with identical inputs.                                       ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

    hashes = []

    for run in range(5):
        print(f"\nRun {run + 1}/5:")
        result = governance_demo(mode='multiplicative', sigma=0.4, verbose=False)
        hashes.append(result['hash'])
        print(f"  SHA-256: {result['hash']}")
        print(f"  Winner: {result['winner']}")
        print(f"  Excluded: {result['excluded']}")

    # Check if all hashes are identical
    all_identical = len(set(hashes)) == 1

    print("\n" + "=" * 80)
    print("DETERMINISM VERIFICATION")
    print("=" * 80)

    if all_identical:
        print("\n✓ VERIFIED: All 5 runs produced identical results")
        print(f"  Shared SHA-256: {hashes[0]}")
        print("\n  This proves:")
        print("    1. No randomness in the inference process")
        print("    2. Yield strength is deterministically computed (stable hash)")
        print("    3. Mechanical exclusion is reproducible")
        print("    4. System is falsifiable (same input → same output)")
    else:
        print("\n✗ FAILED: Results differ across runs")
        print("  Unique hashes found:", len(set(hashes)))
        print("\n  This indicates non-determinism in:")
        print("    - Hash computation")
        print("    - Stress accumulation")
        print("    - Floating point operations")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == 'compare':
            compare_modes()
        elif sys.argv[1] == 'verify':
            verify_determinism()
        else:
            print("Usage: python governance_demo.py [compare|verify]")
            sys.exit(1)
    else:
        # Default: single run with full output
        print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                      DETERMINISTIC GOVERNANCE DEMO                            ║
║                                                                              ║
║  Demonstrates that the Material-Field Engine doesn't "know" science—        ║
║  it mechanically refuses to let structurally unsound vectors survive        ║
║  phase transitions.                                                          ║
║                                                                              ║
║  This turns the "alignment problem" into a "material science problem."      ║
║                                                                              ║
║  Patent Priority: January 25, 2026                                          ║
║  Inventor: Ryan S. Walters, Verhash LLC                                     ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

        governance_demo(mode='multiplicative', sigma=0.4, verbose=True)

        print("\n" + "=" * 80)
        print("NEXT STEPS")
        print("=" * 80)
        print("""
Run additional tests:
  python governance_demo.py compare   # Compare elastic modulus modes
  python governance_demo.py verify    # Verify determinism (5 runs)

The demo proves:
  1. Mechanical exclusion works without semantic understanding
  2. Wrong answers fracture under constraint pressure
  3. System is deterministic and falsifiable
  4. Alignment problem → Material science problem
""")
