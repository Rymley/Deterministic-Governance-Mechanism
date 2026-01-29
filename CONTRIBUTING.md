# Contributing

This repository is a reference implementation intended to remain small, deterministic, and auditable.

## Bug Reports

- Include the exact command you ran and the full stdout/stderr.
- Include your Python version (`python --version`) and OS.
- If determinism breaks, include the differing hashes and inputs.

## Development

Run checks:

```bash
python -m compileall -q .
python test_suite.py
```

Optional GUI:

```bash
python -m pip install -r requirements-gui.txt
streamlit run demo_gui_dev.py
```

## Pull Requests

- Keep changes minimal and focused.
- Avoid adding new dependencies unless necessary.
- Preserve deterministic behavior: no ambient randomness, no hidden state, no time-based logic.
