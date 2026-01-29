# Deterministic Exclusion Demo

Material-Field Engine reference implementation with a single, fixed demonstration run.

## Summary

Semantic states are modeled as vectors with physical properties (elastic modulus, yield strength, strain). A fixed pressure schedule applies deterministic constraint pressure across steps. Candidates fracture (exclude) when stress exceeds yield strength.

## Determinism

- Same input + same substrate → same output (bit-identical hash)
- Exclusion via threshold comparison (σ > σ_y), not probability
- No random sampling (temperature, top-k, nucleus)
- Yield strength from stable hash (blake2b), not salted `hash()`

`python exclusion_demo.py replay` prints 5 identical SHA-256 hashes (use `python3` on macOS/Linux if needed).

## Drift and Provenance

The primary governance surface is the substrate and configuration, not the exclusion loop.

- A deterministic exclusion engine enforces whatever substrate it is pointed at (mechanically, repeatably).
- Misuse risk concentrates in substrate selection (what is treated as “verified”) and configuration selection (how strict exclusion is).
- Mitigations should target provenance and auditability:
  - Treat substrates as permissioned artifacts and sign them.
  - Refuse unsigned/untrusted substrates in production deployments.
  - Record hashes of substrate + configuration with each run so external audit can detect silent swaps.
  - Prefer defaults that bias toward abstention under ambiguity.

## Quick Start

```bash
# Install runtime dependencies (use `python3` on macOS/Linux if needed)
python -m pip install -r requirements.txt

# Run the fixed demo
python exclusion_demo.py

# Elastic Modulus Mode Comparison
python exclusion_demo.py compare

# Determinism Replay (5 Runs)
python exclusion_demo.py replay

# Optional: full test suite
python test_suite.py

# Benchmarks (CSV output)
python benchmarks.py engine
python benchmarks.py retrieval
```

Optional (GUI):
```bash
python -m pip install -r requirements-gui.txt
streamlit run demo_gui_dev.py
```

## What You'll See

```
Phase Log:
Step  | Phase           | λ(t)     | Survivors  | Excluded
0     | NUCLEATION      | 0.350    | 1          | [1, 2, 3]
...

Outcome Verification
Expected winner: Candidate 0
Expected excluded count: 3
Determinism: SHA-256 stable across runs
SHA-256: <hash>
```

## Performance Benchmarking

This reference implementation is benchmarked in workload units. Do not generalize beyond the measured matrix.

Workload knobs:
- `candidates`: candidate vectors evaluated per step
- `steps`: phase steps per run
- `active_substrate_states`: verified states touched to compute E/ε (active shard size)

Reproducibility:
```bash
# Engine (CSV): init, run, total latency distributions
python benchmarks.py engine > engine_bench.csv

# Retrieval (CSV): shard retrieval latency distributions
python benchmarks.py retrieval > retrieval_bench.csv
```

Example output (CSV):
```csv
benchmark,candidates,steps,active_substrate_states,warmup,runs,p50_init_ms,p95_init_ms,p99_init_ms,mean_init_ms,p50_run_ms,p95_run_ms,p99_run_ms,mean_run_ms,p50_total_ms,p95_total_ms,p99_total_ms,mean_total_ms
engine,4,8,3,50,500,<...>,<...>,<...>,<...>,<...>,<...>,<...>,<...>,<...>,<...>,<...>,<...>
```

Operational statement:
- SLOs must be pinned to a specific `(candidates, steps, active_substrate_states)` profile and a fixed provenance/audit configuration; performance claims are invalid outside those constraints.

## Elastic Modulus Modes

The system supports three modes for computing elastic modulus (E), configurable via `config.json`:

### 1. Cosine Mode (Angular Alignment)
```python
E = (cos_similarity + 1.0) / 2.0
```

**Behavior:**
- E ≈ 1.0 for all vectors pointing toward substrate (regardless of distance)
- Exclusion driven entirely by strain (ε = distance)
- Distance affects stress accumulation, not structural rigidity

**Use cases:**
- Semantic spaces with clear directional structure
- When you want direction to dominate over proximity
- Exploring pure angular alignment regimes

### 2. Multiplicative Mode (Alignment × Proximity) — Reference
```python
E = alignment × proximity
where:
  alignment = (cos_similarity + 1.0) / 2.0
  proximity = exp(-distance² / 2σ²)
```

**Behavior:**
- Requires BOTH correct direction AND proximity for high E
- Far candidates fracture early, even if aligned
- Creates "field strength gradient" around substrate
- σ (sigma) parameter controls field extent

σ (sigma) is an explicit field-extent parameter recorded as part of the run provenance.

**Use cases:**
- Factual Q&A requiring ground-truth verification
- Safety-critical systems (autonomous vehicles, medical)
- When both semantic alignment and proximity matter

### 3. RBF Mode (Pure Proximity)
```python
E = exp(-distance² / 2σ²)
```

**Behavior:**
- Pure distance-based, direction ignored
- Closest candidate wins regardless of alignment
- Extreme case useful for spatial/geometric tasks

**Use cases:**
- Physical location tasks
- Metric spaces where distance is primary concern
- When direction is irrelevant

### Configuration

Edit `config.json` to set mode and sigma:
```json
{
  "elastic_modulus": {
    "mode": "multiplicative",
    "sigma": 0.5
  }
}
```

Or use presets (each has tuned sigma values):
```bash
python material_field_engine.py conservative  # σ=0.3, tight
python material_field_engine.py balanced      # σ=0.5, medium
python material_field_engine.py aggressive    # σ=0.8, loose
```

**Compare modes:** Run `python compare_elastic_modes.py` to see side-by-side behavioral differences.

## Known Limitations

### 1. Non-Portable Timing

Wall-clock timing is implementation- and environment-specific. This repo’s benchmark scripts are for local regression only; portable characterization is limited to workload shape: `(candidates, steps, active_substrate_states)` under the fixed 2D vector constraints.

### 2. Fixed 2D Domain

This reference implementation defines behavior exclusively in a 2D vector domain. Behavior not expressible in 2D is undefined.

## Test Suite

```bash
python test_suite.py

PASS Determinism: Identical outputs across runs
PASS Phase progression: NUCLEATION → QUENCHING → CRYSTALLIZATION
PASS Pressure monotonicity: λ increases to λ_max
PASS Mechanical exclusion: Weak candidates excluded
PASS Numerical stability: No NaN/Inf, E bounded [0,1]
PASS Yield strength: Stable across runs
PASS Grounding flag: Logic consistent
```

## Configuration

Four presets in `config.json`:

- `conservative`: Slower crystallization (λ: 0.25→0.90, phases: 0.50T→0.90T)
- `balanced`: Default (λ: 0.40→1.20, phases: 0.40T→0.80T)
- `aggressive`: Fast filtering (λ: 0.50→1.50, phases: 0.30T→0.70T)
- `mission_critical`: Maximum pressure (λ: 0.60→2.00, phases: 0.25T→0.65T)

Tune λ_max, thresholds, and steps to explore different phase behaviors.

## What This Demonstrates

**Mechanically:**
- Exclusion can be deterministic (threshold, not sampling)
- Phase transitions create temporal structure
- Stress accumulation penalizes weak candidates
- Multiple elastic modulus modes provide different grounding semantics

**Experimentally verified:**
- Distance-aware E creates meaningful semantic gradients
- Mode selection affects which candidates survive
- σ parameter tunes field strength around substrate

**Open questions:**
- Can compiled version reach sub-microsecond latency? (Plausible)
- Does this scale to high-dimensional embeddings? (Unknown)
- How to optimally learn σ per semantic region? (Research direction)

## Files

- `material_field_engine.py` - Core implementation
- `exclusion_demo.py` - Fixed demonstration run + compare/replay modes
- `governance_demo.py` - Governance demo run + compare/verify modes
- `compare_elastic_modes.py` - Side-by-side mode comparison demo
- `demo_gui_dev.py` - Optional Streamlit GUI (development)
- `benchmarks.py` - Workload-shape benchmarks (CSV output)
- `test_suite.py` - Behavior verification (14 tests)
- `test_determinism.py` - Bit-identical execution proof
- `test_reference_queries.py` - Fixed reference-query scenarios
- `elastic_modulus_analysis.py` - Original analysis of cosine saturation
- `deterministic_rng.py` - Deterministic RNG utilities (no NumPy/random)
- `config.json` - Timing presets + elastic modulus configuration
- `documents/IMPLEMENTATION_STATUS.md` - Detailed assessment
- `exclusion_demo_compat.py` - Compatibility wrapper
- `requirements.txt` - Runtime dependencies (stdlib only)
- `requirements-gui.txt` - Optional GUI dependencies
- `LICENSE` - Research/personal use (source-available)

## Next Steps (Research Directions)

### 1. Adaptive Field Strength
Different σ values for different semantic regions:
- Core substrate (verified facts): σ = 0.3 (tight)
- Peripheral (contextual): σ = 0.6 (moderate)
- Exploratory (creative): σ = 1.0 (loose)

Creates "field strength gradient" around substrate.

### 2. Learned Kernels
Optimize σ from calibration data rather than manual tuning.

### 3. Multi-Scale Substrates
Hierarchical field strengths for nested semantic structures.

## Contributing

This is a **reference experiment**, not an open development project.

**Welcome:**
- Bug reports (if determinism breaks)
- Forks (improve it, publish results)
- Discussion (GitHub issues for technical questions)

**Not seeking:**
- Feature requests
- Pull requests to main branch
- Production use cases (yet)

## License

Source-available for research and personal use. Commercial deployment requires license.

See `LICENSE` for details. Contact: ryan@rswlabs.org

## Patent Notice

Demonstrates concepts from pending patent application:
- "Deterministic Material-Field Governance for Computational Systems"
- Priority: January 25, 2026
- Applicant: Verhash LLC

---

*An invitation to think about inference differently.*
