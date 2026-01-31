# Deterministic Governance Specification

This document specifies a deterministic governance mechanism designed to eliminate interpretive ambiguity while remaining auditable and defensible.

## Decision Function

All exclusion decisions are defined as a total, binary function over the inequality `σ > σ_y`, evaluated per candidate with no tolerance bands, stochastic modifiers, accumulated state, or adaptive thresholds.

## Vector Domain

System behavior is defined exclusively within a two-dimensional vector domain. All interactions, bindings, and transitions operate solely on direction and magnitude. All transformations MUST preserve these invariants under rotation and scaling.

## Elastic Modulus Modes

Of the three supported Elastic Modulus modes, **Multiplicative** is the reference mode. It derives scalar gradients strictly from vector alignment under composition, ensuring consistent and replayable binding behavior.

Any behavior not expressible within these constraints is undefined and non-compliant.

## Provenance

Provenance is enforced as a hard constraint. Substrates are permissioned, cryptographically signed artifacts with immutable lineage. Evaluation MUST be refused for unsigned, altered, or transitively contaminated inputs.

## Determinism

Drift is mitigated by construction: identical inputs MUST produce identical vector interactions and outcomes within the defined numeric model, with no hidden state, adaptive adjustment, or implicit time dependence. Cross-platform bit-identity requires an explicitly specified numeric model and deterministic serialization (e.g., fixed-point/quantized integers).

## Performance Characterization

Performance characterization is limited to workload-normalized latency expressed strictly in candidate count and step count under fixed two-dimensional vector constraints. Aggregate efficiency claims, hardware-relative metrics, or generalized performance statements are out of scope.
