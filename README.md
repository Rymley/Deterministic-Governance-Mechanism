
---

## `README.md`

```md
# Deterministic Governance Mechanism (Reference Implementation)

> **Status:** Reference implementation (deterministic governance mechanics).  
> **Purpose:** Published to demonstrate feasibility and enable scrutiny.  
> **License:** **Non-commercial only** (personal + research). **Commercial use requires a paid license.** See `LICENSE`.

This repository provides a **deterministic exclusion governance** mechanism: decisions are **mechanical**, not sampled. Given identical inputs, configuration, and substrate, the system produces **bit-identical outputs**.

**Live demo:** https://huggingface.co/spaces/RumleyRum/Deterministic-Governance-Mechanism  
**Production + licensing:** https://verhash.com

---

## Core Invariant

```

Same input + same configuration + same substrate → same output (bit-identical)

````

If two executions diverge, something upstream changed. The system makes divergence visible rather than masking it.

---

## What This Repo Is (and Is Not)

### This repo is
- A **reference implementation** of deterministic governance mechanics
- A reproducible demonstration of **replayability**
- A codebase intended for **personal learning** and **non-commercial research**

### This repo is not
- The production deployment
- A compliance/reporting stack (beyond determinism proofs and logs)
- A grant of commercial rights or permission to deploy in business contexts

For commercial deployment, licensing, and enterprise features: **https://verhash.com**

---

## Verification (Determinism Proof)

Run the same scenario five times:

```bash
python exclusion_demo.py replay
````

Expected output: **five identical SHA-256 hashes**.

Hash is computed as:

```
SHA-256(canonical_input || configuration || substrate_hash || output_decisions)
```

If the hash changes, the computation diverged. If it doesn't, the decision was deterministic.

---

## Mechanism (High Level)

Candidates are stateful objects under constraint pressure. Exclusion occurs when accumulated stress exceeds a fixed yield threshold:

```
σ(t) > σ_y  →  Exclusion
```

No temperature. No sampling. No randomness. Stress accumulates via explicit arithmetic over discrete time steps. Once excluded, a candidate cannot re-enter.

The system implements:

* Deterministic stress accumulation (no entropy sources)
* Cryptographic yield strength (BLAKE2b, no salt)
* Three-phase pressure schedule (nucleation, quenching, crystallization)
* Bit-identical verification (canonical serialization)

All arithmetic is in code. No learned parameters. No hidden state.

---

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
# Mechanical tests verifying invariants
```

---

## Security / Misuse Notes

This engine enforces determinism mechanically; it does not validate the quality of the substrate it is pointed at. Risk concentrates upstream in:

* substrate selection (what is treated as verified)
* configuration selection (how strict exclusion is)

Recommended practice:

* permission and sign substrates
* record configuration + substrate hashes per run
* treat hash divergence as an audit event

---

## Files

* `material_field_engine.py` - Core implementation
* `exclusion_demo.py` - Fixed demonstration run
* `test_suite.py` - Behavior verification tests
* `test_determinism.py` - Bit-identical execution proof
* `config.json` - Timing presets and configuration
* `documents/` - Additional technical documentation

---

## Documentation

* `documents/IMPLEMENTATION_STATUS.md` - Technical architecture and assessment
* `LICENSE` - Usage terms and restrictions
* `SECURITY.md` - Reporting guidelines
* `CONTRIBUTING.md` - Participation guidelines

---

## Commercial Licensing

Commercial use includes (not exhaustive):

* any company/internal use
* integration into products or services
* deployment serving users (paid or free)
* SaaS/API hosting
* use for commercial advantage

Commercial licensing and production deployment: **[https://verhash.com](https://verhash.com)**
Contact: **[ryan@verhash.com](mailto:ryan@verhash.com)** (Verhash LLC)

---

## Patent Notice

This repository demonstrates concepts from pending patent application:

* **Title:** "Deterministic Material-Field Governance for Computational Systems"
* **Priority Date:** January 25, 2026
* **Applicant:** Verhash LLC

---

**Try to break the invariant. If you do, file an issue.**

*An invitation to treat inference as mechanics rather than chance.*

````

---

## `LICENSE` (Non-Commercial Research + Personal)

```md
# Non-Commercial Research & Personal Use License
**Material-Field Governance Reference Implementation**  
Copyright (c) 2026 Ryan S. Walters / Verhash LLC

## 1) Grant of Rights (Non-Commercial Only)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to **use, copy, modify, and distribute** the Software **for non-commercial purposes only**, subject to the terms below.

This is a **source-available** license. It is **not** an OSI “open source” license.

## 2) Permitted Uses (Non-Commercial)

✅ **Personal Use**
- Run the Software for personal learning and experimentation
- Modify for individual non-commercial projects

✅ **Research / Academic Use**
- Study the code and underlying algorithms
- Modify for research experiments
- Publish academic work describing results obtained using the Software
- Use in educational settings (courses, workshops)

✅ **Non-Commercial Redistribution**
- Fork and redistribute the Software (including modifications) **only** under this same license and **only** for non-commercial use
- Submit issues and contributions

## 3) Prohibited Uses (Commercial)

❌ **Commercial Use (Requires a Separate Paid License)**
Commercial use is not permitted under this license. Without a separate commercial license, you may not:
- Use the Software within any business or organization (including internal R&D)
- Deploy in production systems serving users or customers
- Provide the Software as a hosted service (SaaS/API), whether paid or free
- Integrate into products, services, or tooling used for commercial advantage
- Sell, sublicense, or bundle the Software with commercial offerings
- Use the Software as part of paid consulting, contracting, or deliverables

## 4) Conditions

1. **Attribution**
   - You must retain this license text in all copies or substantial portions of the Software.
   - You must credit: **Ryan S. Walters / Verhash LLC**.

2. **Patent Notice (Informational)**
   - The Software demonstrates concepts associated with a pending patent application.
   - Priority date: **January 25, 2026**
   - Title: **"Deterministic Material-Field Governance for Computational Systems"**
   - Applicant: **Verhash LLC**
   - When publishing research results based on this Software, you must include this notice in a reasonable location (e.g., acknowledgments, appendix, or artifact notes).

3. **No Trademark Rights**
   - This license does not grant rights to use Verhash trademarks, names, or logos beyond attribution.

## 5) Warranty Disclaimer

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY.

## 6) Commercial Licensing

If you want commercial use (including internal company use, production deployment, or SaaS/API hosting), you must obtain a separate commercial license.

**Contact:** ryan@verhash.com 
**Organization:** Verhash LLC  
**Website:** https://verhash.com

## 7) Definitions

**“Non-commercial”** means not primarily intended for or directed toward commercial advantage or monetary compensation.

**“Commercial”** includes any use by or for a business/organization, any deployment serving users, any hosting as a service, any integration into a product/service, or any use that supports commercial advantage—whether or not money changes hands.

---

Last updated: March 5, 2026
````

---
