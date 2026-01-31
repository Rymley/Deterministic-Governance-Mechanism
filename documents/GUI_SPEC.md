# Deterministic Exclusion Demo GUI Specification

This document specifies a GUI for monitoring and visualizing the Material-Field Engine reference implementation while preserving the system’s determinism and fixed 2D vector constraints.

## Non-Negotiable Constraints

- The GUI MUST treat the engine as a deterministic black box: identical inputs MUST yield identical displayed outcomes.
- The GUI MUST operate in the native 2D vector domain; it MUST NOT introduce projections to other dimensions.
- Visual transforms are limited to 2D rotation and uniform scaling.
- The GUI MUST surface substrate provenance (artifact identity and integrity signals) and MUST NOT evaluate unsigned or modified substrates when running in a provenance-enforced mode.

## Reference Implementation (Development)

- File: `demo_gui_dev.py`
- Stack: Streamlit + Plotly
- Purpose: rapid inspection of phase logs, exclusion events, and deterministic replay hashes

Run:

```bash
streamlit run demo_gui_dev.py
```

## Views

### 1) Vector Field View (2D)

- Render substrate states and candidate states in 2D.
- Provide optional 2D rotation and uniform scale controls.
- Highlight excluded candidates per step and the surviving candidate (if any).

### 2) Phase Log View

- Step-by-step table: `step`, `phase`, `pressure`, `survivors`, `excluded`.
- Deterministic replay: display a stable run hash derived from structured result data.

### 3) Provenance View

- Show substrate identifier(s) and integrity checks (e.g., hashes, signature status).
- Show configuration snapshot used for the run (mode, sigma, step count, lambda schedule parameters).

## Out of Scope

- Any visualization that depends on dimensionality expansion, learned projections, or adaptive post-processing.
- Any UI behavior that alters engine execution, candidate evaluation, or exclusion decisions.
