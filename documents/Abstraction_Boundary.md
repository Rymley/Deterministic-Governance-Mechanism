---

```markdown
# Abstraction Boundary

This document defines the **abstraction boundary** of the Deterministic Governance Mechanism.

It exists to clearly separate what is **foundational and invariant** from what is
**illustrative, interchangeable, or domain-specific**.

---

## Core Claim

**This repository implements one concrete instantiation of monotonic constraint
reduction; correctness of the mechanism is independent of the specific embedding,
domain, or visualization used.**

The purpose of this project is to demonstrate and specify a *formal substrate*,
not to promote any particular application, model, or interface.

---

## The Substrate

The system is defined algebraically as the 4-tuple:

```

(S, C, ∩, ⊨)

```

Where:

- **S** — State space (set of admissible states)
- **C** — Constraint set (partially ordered under composition)
- **∩** — Intersection operator (monotonic reduction of admissible states)
- **⊨** — Satisfaction relation (logical entailment)

### Structural Invariant

```

∀ c₁, c₂ ∈ C: (S ∩ c₁) ∩ c₂ ⊆ S ∩ c₁

```

Monotonicity is **structural**, not asserted.
Under no sequence of constraint applications can admissibility expand.

This invariant is the foundation of all correctness claims.

---

## Terminal State Semantics

Verification always produces exactly one of the following outcomes:

```

|S_final| = 1  →  Deterministic
|S_final| = 0  →  Abstention
|S_final| > 1  →  NonUniqueness

```

These outcomes are determined solely by **cardinality** of the final admissible set.
No optimization, ranking, scoring, or probability is involved.

---

## Non-Hallucinatory Property

**Theorem**

```

If |S_final| ∈ {0, 1}, the system is provably non-hallucinatory.

```

**Justification**  
A hallucination requires producing content not entailed by constraints.

- When `|S_final| = 1`, the unique state is entailed by construction.
- When `|S_final| = 0`, the system abstains and produces no content.

In neither case can unsupported content be generated.

---

## Path-Dependence

Constraint application order affects the terminal state.

The system is **lawful under composition**, not commutative.

This property is intrinsic to compositional verification systems and appears in:
- Type inference engines
- SMT solvers
- Proof assistants
- Constraint satisfaction systems

Path-dependence is a feature, not a defect.

---

## What Is Demonstrative (Not Foundational)

The following components are **examples**, not requirements:

- Semantic embeddings
- LLM integrations
- Visualization layers
- UI frameworks
- Dimensionality of state space

They exist to *witness* the formalism in a concrete domain.

The same substrate can be instantiated using:
- Graphs, manifolds, or type systems
- Rules, schemas, or proof obligations
- Domains such as access control, planning, verification, or governance

As long as the structural invariant holds, the mechanism remains valid.

---

## Scope Discipline

Changes that **do not** affect correctness:
- Replacing embeddings
- Changing domains
- Rewriting the UI
- Optimizing performance

Changes that **do** affect correctness:
- Violating monotonicity
- Allowing admissibility expansion
- Introducing probabilistic selection
- Generating content when `|S_final| ≠ 1`

This document defines the boundary between those two categories.

---

## Purpose

This abstraction boundary exists to ensure that:

- The mechanism cannot be weakened by reinterpretation
- Applications do not redefine correctness
- Criticism targets the formal model, not the demo
- Extensions remain downstream of the substrate

The goal is not persuasion, but **formal clarity**.

Mathematics does not negotiate.
```
---
