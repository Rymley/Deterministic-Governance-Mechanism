# Mode Comparison Language (Non-Anthropomorphic)

This file provides copy/paste language for describing elastic modulus mode comparisons without implying agency, preference, or intent.

## Template A — Research/Results Writeup

Use this when summarizing logs and outcomes.

> Rewrite this analysis to describe the mechanical differences between direction-only alignment and composite (alignment × proximity) binding. Contrast Cosine Mode’s distance-insensitivity (E derived from angular alignment only) with Multiplicative Mode’s composite gate (E = alignment_term × proximity_term). Describe how σ (sigma) controls the proximity bandwidth, contracting or expanding the effective binding radius. Avoid subjective framing (e.g., “prefers”); describe outcomes in terms of exclusion/fracture under σ > σ_y.

## Template B — Technical Definition

Use this when defining kernels in a README or spec.

> Define the three elastic modulus kernels:
> - Cosine: E depends on direction (alignment) only; distance/magnitude does not reduce E.
> - Multiplicative: E = alignment_term × proximity_term; a composite gate that requires both alignment and proximity.
> - RBF: E depends on proximity only.
> Keep descriptions operational and deterministic. Treat σ (sigma) as an explicit configuration parameter recorded in run provenance; do not present ranges as normative policy.

## Template C — System Rationale (Why Kernel Choice Matters)

Use this when explaining why the kernels produce different outcomes.

> Reframe the comparison as a property of the kernel: Cosine Mode can assign high E to candidates that are directionally aligned but distant, because E is insensitive to magnitude. Multiplicative Mode reduces E as distance grows (via the proximity term), so distant candidates are more likely to fracture under constraint pressure. Emphasize that this is a mechanical exclusion condition, not a ranking “preference.”

## Language Rules

- Refer to Cosine as “direction-only” or “distance-insensitive” (or “saturated” when illustrating saturation).
- Refer to Multiplicative as “composite” or “gated”.
- Do not claim “truth”, “understanding”, “preference”, or “intent”. Use “excludes/fractures” instead.

## Determinism / Drift Claims (Safe Wording)

Avoid absolute “zero drift” claims unless the numeric model is fully specified and enforced.

Preferred phrasing:
- “Bit-identical across repeated runs with identical inputs in the same implementation and pinned environment.”
- “Cross-platform bit-identity requires an explicit numeric model (e.g., fixed-point/quantized integers) and deterministic serialization.”
