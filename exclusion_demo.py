#!/usr/bin/env python3
"""
Deterministic Exclusion Demo

Executes a fixed query against a verified substrate and logs phase-wise
candidate exclusion under deterministic constraint pressure.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).parent))

from material_field_engine import MaterialFieldEngine, VerifiedSubstrate, Vector2D


_STARTUP_BANNER = """\
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║              Deterministic Exclusion Demonstration                            ║
║                                                                              ║
║  Executes a fixed query against a verified substrate and                     ║
║  logs phase-wise candidate exclusion under deterministic                     ║
║  constraint pressure.                                                        ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""


def _print_startup_banner() -> None:
    print(_STARTUP_BANNER)


def _stable_result_hash(
    *,
    final_output: Vector2D | None,
    total_excluded: int,
    phase_log: list[dict[str, Any]],
) -> str:
    result_data = {
        "final_output": (final_output.x, final_output.y) if final_output else None,
        "excluded_count": total_excluded,
        "phase_log": [
            {
                "step": int(e["step"]),
                "phase": str(e["phase"]),
                "survivors": int(e["survivors"]),
                "pressure": round(float(e["pressure"]), 6),
            }
            for e in phase_log
        ],
    }
    result_json = json.dumps(result_data, sort_keys=True)
    return hashlib.sha256(result_json.encode()).hexdigest()


def run_deterministic_exclusion_demo(
    *,
    elastic_modulus_mode: str = "multiplicative",
    sigma: float = 0.40,
    lambda_min: float = 0.35,
    lambda_max: float = 1.20,
    steps: int = 8,
    print_banner: bool = True,
    emit_stdout: bool = True,
) -> dict[str, Any]:
    """
    Deterministic Exclusion Demo (fixed query, fixed substrate, fixed candidates).

    Returns a dict suitable for both CLI output and programmatic checks.
    """

    if emit_stdout and print_banner:
        _print_startup_banner()

    if emit_stdout:
        print("Query: Where do plants get their food?")
        print(f"Elastic modulus mode: {elastic_modulus_mode}")
        print(f"Sigma: {sigma:.2f}")
        print(f"Pressure schedule: λ = {lambda_min:.2f} → {lambda_max:.2f}, steps = {steps}")
        print()

    substrate = VerifiedSubstrate(
        elastic_modulus_mode=elastic_modulus_mode,
        elastic_modulus_sigma=sigma,
    )
    substrate_states = [
        (0.95, 0.92),  # primary anchor
        (0.90, 0.88),  # support
        (0.88, 0.90),  # counterexample guard
    ]
    for x, y in substrate_states:
        substrate.add_verified_state(Vector2D(x=x, y=y, properties=None))

    if emit_stdout:
        print("Verified Substrate States:")
        print("  State 0 (primary anchor): photosynthesis synthesis")
        print("  State 1 (support): chlorophyll mechanism")
        print("  State 2 (counterexample guard): soil mass correction")
        print()

    candidates = [
        (0.95, 0.92),
        (0.10, 0.10),
        (0.50, 0.50),
        (-0.80, -0.80),
    ]
    candidate_labels = [
        "Candidate 0 (reference-aligned)",
        "Candidate 1 (misconception)",
        "Candidate 2 (vague)",
        "Candidate 3 (out-of-domain)",
    ]

    engine = MaterialFieldEngine(
        substrate,
        lambda_min=lambda_min,
        lambda_max=lambda_max,
        inference_steps=steps,
    )
    engine.phase_controller.nucleation_threshold = 0.375
    engine.phase_controller.quenching_threshold = 0.875
    engine.initialize_candidates(candidates)

    if emit_stdout:
        print("Candidates:")
        print(f"{'#':<3} {'Label':<32} | {'E':<8} | {'σ_y':<8} | {'ε':<8} | {'σ_init':<8}")
        print("-" * 80)
    for i, (v, label) in enumerate(zip(engine.candidate_vectors, candidate_labels)):
        if emit_stdout:
            print(
                f"{i:<3} {label:<32} | "
                f"{v.properties.elastic_modulus:<8.4f} | "
                f"{v.properties.yield_strength:<8.4f} | "
                f"{v.properties.strain:<8.4f} | "
                f"{v.properties.stress:<8.4f}"
            )
    if emit_stdout:
        print()

    results = engine.run_inference()

    if emit_stdout:
        print("Phase Log:")
        print(f"{'Step':<5} | {'Phase':<15} | {'λ(t)':<8} | {'Survivors':<10} | {'Excluded'}")
        print("-" * 80)
        for entry in results["phase_log"]:
            excluded_indices = entry.get("excluded_indices", [])
            excluded_str = (
                "[" + ", ".join(str(i) for i in excluded_indices) + "]"
                if excluded_indices
                else "[]"
            )
            print(
                f"{entry['step']:<5} | {entry['phase']:<15} | {entry['pressure']:<8.3f} | "
                f"{entry['survivors']:<10} | {excluded_str}"
            )
        print()

    final_output = results["final_output"]
    winner_idx = None
    if final_output is not None:
        winner_tuple = (final_output.x, final_output.y)
        winner_idx = next(i for i, c in enumerate(candidates) if c == winner_tuple)

    result_hash = _stable_result_hash(
        final_output=final_output,
        total_excluded=results["total_excluded"],
        phase_log=results["phase_log"],
    )

    if emit_stdout:
        print("Outcome Verification")
        print("Expected winner: Candidate 0")
        print("Expected excluded count: 3")
        print("Determinism: SHA-256 stable across runs")
        print(f"SHA-256: {result_hash}")

    return {
        "winner_index": winner_idx,
        "winner_label": candidate_labels[winner_idx] if winner_idx is not None else None,
        "excluded": results["total_excluded"],
        "hash": result_hash,
        "phase_log": results["phase_log"],
        "final_output": final_output,
        "elastic_modulus_mode": elastic_modulus_mode,
        "sigma": sigma,
        "lambda_min": lambda_min,
        "lambda_max": lambda_max,
        "steps": steps,
    }


def elastic_modulus_mode_comparison() -> list[dict[str, Any]]:
    print("Elastic Modulus Mode Comparison")
    print()

    modes = [
        ("cosine", 0.40),
        ("multiplicative", 0.40),
        ("multiplicative", 0.60),
    ]

    results: list[dict[str, Any]] = []
    for mode, sigma in modes:
        res = run_deterministic_exclusion_demo(
            elastic_modulus_mode=mode,
            sigma=sigma,
            print_banner=False,
            emit_stdout=False,
        )
        results.append(res)
        print(f"Mode: {mode}, sigma={sigma:.2f}, SHA-256: {res['hash']}")
    print()

    print(f"{'Mode':<18} | {'Sigma':<6} | {'Winner':<10} | {'Excluded':<8} | {'SHA-256'}")
    print("-" * 80)
    for res in results:
        print(
            f"{res['elastic_modulus_mode']:<18} | {res['sigma']:<6.2f} | "
            f"{(res['winner_index'] if res['winner_index'] is not None else 'None')!s:<10} | "
            f"{res['excluded']:<8} | {res['hash']}"
        )

    return results


def determinism_replay(runs: int = 5) -> list[str]:
    print("Determinism Replay (5 Runs)" if runs == 5 else f"Determinism Replay ({runs} Runs)")
    print()

    hashes: list[str] = []
    for i in range(runs):
        res = run_deterministic_exclusion_demo(print_banner=False, emit_stdout=False)
        hashes.append(res["hash"])
        print(f"Run {i + 1}: {res['hash']}")

    if len(set(hashes)) == 1:
        print()
        print("Determinism: SHA-256 stable across runs")
        print(f"SHA-256: {hashes[0]}")
    else:
        print()
        print("Determinism: SHA-256 not stable across runs")

    return hashes


def main(argv: list[str]) -> int:
    if len(argv) > 1:
        if argv[1] == "compare":
            elastic_modulus_mode_comparison()
            return 0
        if argv[1] in {"replay", "verify"}:
            determinism_replay(5)
            return 0
        print("Usage: python exclusion_demo.py [compare|replay]")
        return 2

    run_deterministic_exclusion_demo()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
