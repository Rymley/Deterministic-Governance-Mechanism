````md
# Deterministic Governance Mechanism

A reference implementation of deterministic exclusion: a governance engine where decisions are produced by a hard mechanical threshold, not probabilistic ranking or sampling. Given identical inputs, configuration, and runtime substrate, the engine produces bit-identical outputs. This repository is not a policy proposal or moderation system; it is a mechanism, a minimal and inspectable experiment showing that exclusion decisions can be causal, replayable, and mechanically auditable.

## Overview

The system models candidates as stateful objects subject to deterministic constraint pressure over time. Each candidate accumulates stress, and exclusion occurs only when accumulated stress exceeds a fixed yield threshold. Once excluded, a candidate cannot re-enter; history is part of the state. There is no randomness, temperature, ranking, sampling, or learned scoring. The engine behaves like a material system under load: given the same initial conditions and pressure schedule, the same fractures occur every time.

## Core Invariant

The fundamental invariant enforced by the engine is:

```text
Same input + same configuration + same substrate → same output (bit-identical)
````

If two executions produce different outputs, then something upstream has changed: the inputs, the configuration, the substrate, or the runtime environment. The engine makes this divergence visible rather than hiding it behind probability.

## Mechanical Model

Each candidate i has state variables for accumulated stress σ_i(t), yield strength σ_y,i, and fracture state (intact or excluded). Exclusion is a one-way state transition governed by a hard threshold:

```text
σ_i(t) > σ_y,i  →  candidate fractures (excluded)
```

Once fractured, a candidate cannot re-enter. There is no decay, annealing, or reset.

Stress evolves deterministically across discrete steps:

```text
σ_i(k+1) = σ_i(k) + Δσ_i(k)
```

All increments Δσ are computed from explicit deterministic functions defined in code.

## Constraint Pressure and Phases

Constraint pressure is applied via a deterministic schedule λ(k) partitioned into three phases. These phases are explicit intervals in the pressure schedule and are recorded in run provenance. Nucleation is the initial pressure ramp that filters candidates failing basic structural constraints. Quenching is higher-frequency pressure application that amplifies contradictions and internal inconsistencies. Crystallization is sustained pressure that verifies stability under continued constraints. Phase boundaries, pressure values, and step counts are fixed by configuration and included in the provenance hash.

```text
λ(t)  Constraint Pressure
│
│        ┌────────────────── Crystallization ──────────────────┐
│      ┌─┘
│    ┌─┘   Quenching
│  ┌─┘
│┌─┘ Nucleation
└─────────────────────────────────────────────────────────────── t
```

## Elastic Modulus and Stress Accumulation

At each step, stress increments are computed from measurable terms such as alignment and proximity to a verified substrate. The engine supports multiple elastic modulus formulations (for example cosine, multiplicative, and RBF), selectable via configuration. Changing the elastic modulus formulation changes how stress accumulates, not whether the process is deterministic. The exclusion rule remains identical across modes. All arithmetic is explicit and inspectable in code, and no learned parameters are involved.

## Yield Strength

Yield strength σ_y is derived deterministically using a cryptographic hash (BLAKE2b). Language-dependent or salted hash functions are explicitly avoided. Given the same candidate identity and configuration, yield strength is stable across runs.

## Determinism and Replay

The engine does not rely on entropy sources, random seeds, sampling strategies, or hidden randomness. Canonicalization is applied where necessary through deterministic serialization, stable hashing, and explicit configuration capture. The repository includes a replay mode that executes the same run multiple times and prints identical SHA-256 hashes.

## Bit-Identical Verification

Each run produces a reproducibility artifact derived from a canonical hash over input, configuration, substrate hash, and output report:

```text
H = SHA-256(canonical_input || config || substrate_hash || output)
```

If two users produce different hashes, then the computation has diverged. The engine does not attempt to mask or compensate for this divergence. This shifts the trust surface from “did the model behave well?” to “did the computation remain invariant?”

## Provenance and Misuse Risk

The engine enforces determinism mechanically; it does not validate the quality of the substrate it is pointed at. Misuse risk concentrates upstream in substrate selection (what is treated as verified) and configuration selection (how strict exclusion is). Mitigations therefore target provenance and auditability: substrates should be permissioned and signed, configuration and substrate hashes should be recorded with each run, silent swaps must be detectable via hash divergence, and defaults should bias toward abstention under ambiguity.

## What This Is and Is Not

This repository demonstrates that exclusion can be deterministic, decisions can be replayed and audited, and governance logic can be mechanical rather than probabilistic. It does not claim optimality, fairness, scalability to high-dimensional embeddings, or readiness for production deployment; those remain open research questions.

## Status

This is a reference experiment, not an open development project. The code is intended to be read, tested, forked, and reasoned about, and independent analysis or alternative implementations are welcome.

## License and Notice

Source-available for research and personal use. Commercial deployment requires a separate license. Concepts demonstrated are covered by a pending patent application.

*An invitation to treat inference as mechanics rather than chance.*

```
::contentReference[oaicite:0]{index=0}
```
