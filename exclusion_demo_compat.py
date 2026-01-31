#!/usr/bin/env python3
"""
Compatibility wrapper.

Use `exclusion_demo.py` / `run_deterministic_exclusion_demo()`.
"""

import sys

from exclusion_demo import (
    run_deterministic_exclusion_demo,
    elastic_modulus_mode_comparison,
    determinism_replay,
)


def governance_demo(mode: str = "multiplicative", sigma: float = 0.4, verbose: bool = True):
    # `verbose` retained for signature compatibility; output is always structured.
    return run_deterministic_exclusion_demo(
        elastic_modulus_mode=mode,
        sigma=sigma,
        print_banner=verbose,
        emit_stdout=verbose,
    )


def compare_modes():
    return elastic_modulus_mode_comparison()


def verify_determinism():
    return determinism_replay(5)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "compare":
            compare_modes()
            raise SystemExit(0)
        if sys.argv[1] in {"verify", "replay"}:
            verify_determinism()
            raise SystemExit(0)
        print("Usage: python exclusion_demo.py [compare|replay]")
        raise SystemExit(2)

    run_deterministic_exclusion_demo()
