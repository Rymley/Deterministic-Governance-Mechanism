# Deterministic Governance Mechanism

Probabilistic systems cannot be audited. If a decision changes between runs with identical inputs, the reasoning chain is non-reproducible, and post-hoc explanation is speculation.

This is a reference implementation of deterministic exclusion: a governance layer where decisions are mechanical, not sampled. Given identical inputs, configuration, and substrate, the system produces bit-identical outputs.

**[Try the live demo →](https://huggingface.co/spaces/RumleyRum/Deterministic-Governance-Mechanism)**

## Core Invariant

```
Same input + same configuration + same substrate → same output (bit-identical)
```

If two executions diverge, something upstream changed. The system makes divergence visible rather than masking it.

## Verification

Run the same scenario five times:

```bash
python exclusion_demo.py replay
```

Output: Five identical SHA-256 hashes.

```
SHA-256(canonical_input || configuration || substrate_hash || output_decisions)
```

If the hash changes, the computation diverged. If it doesn't, the decision was deterministic.

## Mechanism

Candidates are stateful objects under constraint pressure. Exclusion occurs when accumulated stress exceeds a fixed yield threshold:

```
σ(t) > σ_y  →  Exclusion
```

No temperature. No sampling. No randomness. Stress accumulates via explicit arithmetic over discrete time steps. Once excluded, a candidate cannot re-enter.

The system implements:
- Deterministic stress accumulation (no entropy sources)
- Cryptographic yield strength (BLAKE2b, no salt)
- Three-phase pressure schedule (nucleation, quenching, crystallization)
- Bit-identical verification (canonical serialization)

All arithmetic is in code. No learned parameters. No hidden state.

## Quick Start

```bash
git clone https://github.com/Rymley/Deterministic-Governance-Mechanism
cd Deterministic-Governance-Mechanism
pip install -r requirements.txt
python exclusion_demo.py
```

**Prove determinism:**
```bash
python exclusion_demo.py replay
# Runs 5 times - prints identical SHA-256 hashes
```

**Compare modes:**
```bash
python exclusion_demo.py compare
# Shows behavioral differences across elastic modulus modes
```

**Run full test suite:**
```bash
python test_suite.py
# 14 mechanical tests verifying invariants
```

## What This Is

An experiment showing exclusion can be:
- **Deterministic** (same inputs → same outputs)
- **Replayable** (hash proves invariance)
- **Mechanical** (threshold, not probability)

## What This Is Not

- A production system
- A claim about optimality or fairness
- A solution to high-dimensional scaling (open question)
- A validation of substrate quality (garbage in, deterministic garbage out)

## Provenance and Misuse

The engine enforces determinism mechanically; it does not validate the quality of the substrate it is pointed at. Misuse risk concentrates upstream in substrate selection (what is treated as verified) and configuration selection (how strict exclusion is).

Mitigations target provenance and auditability:
- Substrates should be permissioned and signed
- Configuration and substrate hashes recorded with each run
- Silent swaps detectable via hash divergence
- Defaults bias toward abstention under ambiguity

## Files

- `material_field_engine.py` - Core implementation
- `exclusion_demo.py` - Fixed demonstration run
- `test_suite.py` - Behavior verification (14 tests)
- `test_determinism.py` - Bit-identical execution proof
- `config.json` - Timing presets and configuration
- `documents/` - Detailed technical documentation

## Documentation

- [Implementation Status](documents/IMPLEMENTATION_STATUS.md) - Technical architecture and assessment
- [License](LICENSE) - Usage terms and restrictions
- [Security Policy](SECURITY.md) - Reporting guidelines
- [Contributing](CONTRIBUTING.md) - Participation guidelines

## Production Use

This is a reference implementation demonstrating core mechanics.

For production-ready deployment with enterprise features → **[verhash.com](https://verhash.com)**

## Commercial Partnerships

Interested in technology licensing, partnerships, or investment:

**Contact:** ryan@verhash.net  
**Organization:** Verhash LLC

## Patent Notice

Demonstrates concepts from pending patent application:
- **Title:** "Deterministic Material-Field Governance for Computational Systems"
- **Priority Date:** January 25, 2026
- **Applicant:** Verhash LLC

## License

**Research and Personal Use:** Open source  
**Commercial Deployment:** Requires separate license

See [LICENSE](LICENSE) for full terms.

---

**Try to break the invariant. If you do, file an issue.**

*An invitation to treat inference as mechanics rather than chance.*
