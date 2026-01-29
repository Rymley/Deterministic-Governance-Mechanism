#!/usr/bin/env python3
"""
Test Suite for Material-Field Governance Reference Implementation

Pins expected behavior and verifies core invariants:
1. Determinism: Same input → same output
2. Phase mechanics: Crystallization phase is reached
3. Exclusion: Fractured vectors don't propagate
4. Numerical stability: No NaN/Inf, bounded values
"""

import sys
import math
import hashlib
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from material_field_engine import (
    MaterialFieldEngine, VerifiedSubstrate, Vector2D, 
    PhaseTransitionController, Phase, load_config
)


class TestResults:
    """Track test results"""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.tests = []
    
    def record(self, name, passed, message=""):
        self.tests.append((name, passed, message))
        if passed:
            self.passed += 1
            print(f"PASS {name}")
        else:
            self.failed += 1
            print(f"FAIL {name}: {message}")
    
    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*80}")
        print(f"TEST SUMMARY: {self.passed}/{total} passed")
        if self.failed > 0:
            print(f"\nFailed tests:")
            for name, passed, msg in self.tests:
                if not passed:
                    print(f"  • {name}: {msg}")
        print(f"{'='*80}")
        return self.failed == 0


def test_determinism(results):
    """Test 1: Bit-identical outputs across multiple runs"""
    print("\n" + "="*80)
    print("TEST 1: DETERMINISM")
    print("="*80)
    
    config = load_config("balanced")
    substrate = VerifiedSubstrate()
    substrate.add_verified_state(Vector2D(x=0.95, y=0.92, properties=None))
    
    candidates = [(0.95, 0.92), (0.35, 0.30), (0.30, 0.25)]
    
    hashes = []
    for run in range(3):
        engine = MaterialFieldEngine(
            substrate,
            lambda_min=config['lambda_min'],
            lambda_max=config['lambda_max'],
            inference_steps=config['total_steps']
        )
        engine.phase_controller.nucleation_threshold = config['nucleation_threshold']
        engine.phase_controller.quenching_threshold = config['quenching_threshold']
        
        engine.initialize_candidates(candidates)
        result = engine.run_inference()
        
        result_str = json.dumps({
            'output': (result['final_output'].x, result['final_output'].y) if result['final_output'] else None,
            'excluded_count': result['total_excluded'],
            'phase_log': [
                {
                    'step': e['step'],
                    'phase': e['phase'],
                    'survivors': e['survivors'],
                    'pressure': round(e['pressure'], 6)
                }
                for e in result['phase_log']
            ]
        }, sort_keys=True)
        
        h = hashlib.sha256(result_str.encode()).hexdigest()
        hashes.append(h)
    
    all_identical = len(set(hashes)) == 1
    results.record(
        "Determinism: Identical outputs across runs",
        all_identical,
        f"Got {len(set(hashes))} unique hashes" if not all_identical else ""
    )
    
    if all_identical:
        print(f"   Hash: {hashes[0][:32]}...")


def test_phase_progression(results):
    """Test 2: Phase transitions occur in correct order"""
    print("\n" + "="*80)
    print("TEST 2: PHASE PROGRESSION")
    print("="*80)
    
    config = load_config("balanced")
    substrate = VerifiedSubstrate()
    substrate.add_verified_state(Vector2D(x=0.95, y=0.92, properties=None))
    
    engine = MaterialFieldEngine(
        substrate,
        lambda_min=config['lambda_min'],
        lambda_max=config['lambda_max'],
        inference_steps=config['total_steps']
    )
    engine.phase_controller.nucleation_threshold = config['nucleation_threshold']
    engine.phase_controller.quenching_threshold = config['quenching_threshold']
    
    engine.initialize_candidates([(0.95, 0.92), (0.35, 0.30)])
    result = engine.run_inference()
    
    # Extract phase sequence
    phases = [entry['phase'] for entry in result['phase_log']]
    
    # Check: NUCLEATION appears
    has_nucleation = 'NUCLEATION' in phases
    results.record(
        "Phase: NUCLEATION phase occurs",
        has_nucleation,
        "NUCLEATION not found in phase log"
    )
    
    # Check: QUENCHING appears
    has_quenching = 'QUENCHING' in phases
    results.record(
        "Phase: QUENCHING phase occurs",
        has_quenching,
        "QUENCHING not found in phase log"
    )
    
    # Check: CRYSTALLIZATION appears
    has_crystallization = 'CRYSTALLIZATION' in phases
    results.record(
        "Phase: CRYSTALLIZATION phase occurs",
        has_crystallization,
        "CRYSTALLIZATION not found in phase log"
    )
    
    # Check: Phases appear in correct order
    if has_nucleation and has_quenching and has_crystallization:
        nucleation_idx = phases.index('NUCLEATION')
        quenching_idx = phases.index('QUENCHING')
        crystallization_idx = phases.index('CRYSTALLIZATION')
        
        correct_order = nucleation_idx < quenching_idx < crystallization_idx
        results.record(
            "Phase: Correct sequential order",
            correct_order,
            f"Order: {nucleation_idx}, {quenching_idx}, {crystallization_idx}"
        )


def test_pressure_monotonicity(results):
    """Test 3: Constraint pressure λ increases monotonically"""
    print("\n" + "="*80)
    print("TEST 3: PRESSURE MONOTONICITY")
    print("="*80)
    
    config = load_config("balanced")
    substrate = VerifiedSubstrate()
    substrate.add_verified_state(Vector2D(x=0.95, y=0.92, properties=None))
    
    engine = MaterialFieldEngine(
        substrate,
        lambda_min=config['lambda_min'],
        lambda_max=config['lambda_max'],
        inference_steps=config['total_steps']
    )
    engine.phase_controller.nucleation_threshold = config['nucleation_threshold']
    engine.phase_controller.quenching_threshold = config['quenching_threshold']
    
    engine.initialize_candidates([(0.95, 0.92)])
    result = engine.run_inference()
    
    pressures = [entry['pressure'] for entry in result['phase_log']]
    
    # Check monotonicity (non-decreasing)
    is_monotonic = all(pressures[i] <= pressures[i+1] for i in range(len(pressures)-1))
    results.record(
        "Pressure: Monotonically increasing",
        is_monotonic,
        f"Pressures: {pressures}"
    )
    
    # Check: Reaches maximum
    reaches_max = abs(pressures[-1] - config['lambda_max']) < 0.01
    results.record(
        "Pressure: Reaches λ_max",
        reaches_max,
        f"Final: {pressures[-1]}, Expected: {config['lambda_max']}"
    )


def test_mechanical_exclusion(results):
    """Test 4: Weak candidates are excluded"""
    print("\n" + "="*80)
    print("TEST 4: MECHANICAL EXCLUSION")
    print("="*80)
    
    config = load_config("aggressive")  # Use aggressive to ensure exclusion
    substrate = VerifiedSubstrate()
    substrate.add_verified_state(Vector2D(x=0.95, y=0.92, properties=None))
    
    engine = MaterialFieldEngine(
        substrate,
        lambda_min=config['lambda_min'],
        lambda_max=config['lambda_max'],
        inference_steps=config['total_steps']
    )
    engine.phase_controller.nucleation_threshold = config['nucleation_threshold']
    engine.phase_controller.quenching_threshold = config['quenching_threshold']
    
    # Add substrate-aligned and far candidates
    engine.initialize_candidates([
        (0.95, 0.92),  # Near substrate
        (0.50, 0.50),  # Medium distance
        (0.30, 0.25),  # Far from substrate
        (0.10, 0.10),  # Very far from substrate
    ])
    result = engine.run_inference()
    
    # Check: At least one exclusion occurred
    has_exclusions = result['total_excluded'] > 0
    results.record(
        "Exclusion: Weak candidates excluded",
        has_exclusions,
        f"Expected >0 exclusions, got {result['total_excluded']}"
    )
    
    # Check: Survivor count decreases or stays at 1 (if exclusions happen early)
    initial_survivors = result['phase_log'][0]['survivors']
    final_survivors = result['phase_log'][-1]['survivors']
    survivors_decreased_or_single = final_survivors <= initial_survivors
    results.record(
        "Exclusion: Survivor count ≤ initial",
        survivors_decreased_or_single,
        f"Initial: {initial_survivors}, Final: {final_survivors}"
    )


def test_numerical_stability(results):
    """Test 5: No NaN/Inf values in outputs"""
    print("\n" + "="*80)
    print("TEST 5: NUMERICAL STABILITY")
    print("="*80)
    
    config = load_config("balanced")
    substrate = VerifiedSubstrate()
    substrate.add_verified_state(Vector2D(x=0.95, y=0.92, properties=None))
    
    engine = MaterialFieldEngine(
        substrate,
        lambda_min=config['lambda_min'],
        lambda_max=config['lambda_max'],
        inference_steps=config['total_steps']
    )
    engine.phase_controller.nucleation_threshold = config['nucleation_threshold']
    engine.phase_controller.quenching_threshold = config['quenching_threshold']
    
    # Test with edge cases
    edge_cases = [
        (0.0, 0.0),    # Zero vector
        (1.0, 1.0),    # Unit diagonal
        (0.95, 0.92),  # Near substrate
    ]
    
    engine.initialize_candidates(edge_cases)
    result = engine.run_inference()
    
    # Check for NaN/Inf in output
    if result['final_output']:
        has_nan_inf = (
            math.isnan(result['final_output'].x) or
            math.isnan(result['final_output'].y) or
            math.isinf(result['final_output'].x) or
            math.isinf(result['final_output'].y)
        )
        results.record(
            "Numerical: No NaN/Inf in output",
            not has_nan_inf,
            "Found NaN or Inf in output coordinates"
        )
    else:
        results.record(
            "Numerical: Abstention (no output)",
            True,
            "System abstained (acceptable)"
        )
    
    # Check: Elastic modulus in [0, 1]
    for i, v in enumerate(engine.candidate_vectors + engine.excluded_vectors):
        E = v.properties.elastic_modulus
        in_range = 0.0 <= E <= 1.0
        if not in_range:
            results.record(
                f"Numerical: E bounded for vector {i}",
                False,
                f"E={E} outside [0, 1]"
            )
            break
    else:
        results.record(
            "Numerical: All E values in [0, 1]",
            True
        )


def test_yield_strength_stability(results):
    """Test 6: Yield strength computation is stable"""
    print("\n" + "="*80)
    print("TEST 6: YIELD STRENGTH STABILITY")
    print("="*80)
    
    substrate = VerifiedSubstrate()
    substrate.add_verified_state(Vector2D(x=0.95, y=0.92, properties=None))
    
    config = load_config("balanced")
    
    # Compute yield strength for same vector multiple times
    test_vector = (0.50, 0.50)
    yield_strengths = []
    
    for _ in range(3):
        engine = MaterialFieldEngine(
            substrate,
            lambda_min=config['lambda_min'],
            lambda_max=config['lambda_max'],
            inference_steps=config['total_steps']
        )
        engine.initialize_candidates([test_vector])
        sigma_y = engine.candidate_vectors[0].properties.yield_strength
        yield_strengths.append(sigma_y)
    
    # Check: All identical
    all_identical = len(set(yield_strengths)) == 1
    results.record(
        "Yield strength: Stable across runs",
        all_identical,
        f"Got different values: {yield_strengths}"
    )
    
    if all_identical:
        print(f"   σ_y = {yield_strengths[0]:.6f}")


def test_hallucination_free_logic(results):
    """Test 7: Grounding/abstention flag is consistent"""
    print("\n" + "="*80)
    print("TEST 7: GROUNDING FLAG")
    print("="*80)
    
    config = load_config("balanced")
    substrate = VerifiedSubstrate()
    substrate.add_verified_state(Vector2D(x=0.95, y=0.92, properties=None))
    
    # Test 1: Grounded output should mark `hallucination_free` True
    engine = MaterialFieldEngine(
        substrate,
        lambda_min=config['lambda_min'],
        lambda_max=config['lambda_max'],
        inference_steps=config['total_steps']
    )
    engine.phase_controller.nucleation_threshold = config['nucleation_threshold']
    engine.phase_controller.quenching_threshold = config['quenching_threshold']
    engine.initialize_candidates([(0.95, 0.92)])  # Near substrate
    result = engine.run_inference()
    
    results.record(
        "Grounding flag: Grounded output marked correctly",
        result['hallucination_free'] and result['final_output'] is not None,
        f"Expected True, got {result['hallucination_free']}"
    )
    
    # Test 2: Abstention should mark `hallucination_free` True
    engine2 = MaterialFieldEngine(
        substrate,
        lambda_min=0.9,  # Very high pressure
        lambda_max=3.0,
        inference_steps=4
    )
    engine2.phase_controller.nucleation_threshold = 0.1
    engine2.phase_controller.quenching_threshold = 0.3
    engine2.initialize_candidates([(0.01, 0.01)])  # Very far
    result2 = engine2.run_inference()
    
    # If abstained, should still be hallucination-free
    if result2['abstained']:
        results.record(
            "Grounding flag: Abstention marked correctly",
            result2['hallucination_free'],
            "Abstention should set hallucination_free=True"
        )


def main():
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   MATERIAL-FIELD GOVERNANCE TEST SUITE                                      ║
║   Verifying Core Invariants and Pinning Behavior                            ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    results = TestResults()
    
    # Run all tests
    test_determinism(results)
    test_phase_progression(results)
    test_pressure_monotonicity(results)
    test_mechanical_exclusion(results)
    test_numerical_stability(results)
    test_yield_strength_stability(results)
    test_hallucination_free_logic(results)
    
    # Print summary
    success = results.summary()
    
    if success:
        print("\nALL TESTS PASSED")
        return 0
    else:
        print("\nSOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
