#!/usr/bin/env python3
"""
Deterministic Exclusion Demo GUI (Development Build)
FastAPI + Streamlit implementation for rapid prototyping

Usage:
    streamlit run demo_gui_dev.py
"""

try:
    import streamlit as st
    import plotly.graph_objects as go
    import plotly.express as px
except ModuleNotFoundError as exc:
    missing = str(exc).split("No module named ", 1)[-1].strip("'\"")
    print(f"Missing optional GUI dependency: {missing}")
    print("Install GUI deps: python -m pip install -r requirements-gui.txt")
    print("Run the GUI via:  streamlit run demo_gui_dev.py")
    raise SystemExit(2)

try:
    from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx
except Exception:
    get_script_run_ctx = None

if get_script_run_ctx is not None and get_script_run_ctx() is None:
    print("This file is a Streamlit app and must be run with Streamlit:")
    print("  streamlit run demo_gui_dev.py")
    raise SystemExit(2)
import time
import hashlib
import json
from pathlib import Path
from collections import deque

# Import engine components
import sys
sys.path.insert(0, str(Path(__file__).parent))

from material_field_engine import (
    VerifiedSubstrate, Vector2D, MaterialFieldEngine, load_config
)
from exclusion_demo import run_deterministic_exclusion_demo


# ============================================================================
# Streamlit Page Configuration
# ============================================================================

st.set_page_config(
    page_title="Deterministic Exclusion Demo",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for monospace hash display
st.markdown("""
<style>
    .hash-display {
        font-family: 'Courier New', monospace;
        font-size: 14px;
        background-color: #1e1e1e;
        color: #00ff00;
        padding: 10px;
        border-radius: 5px;
        word-break: break-all;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Session State Initialization
# ============================================================================

if 'results' not in st.session_state:
    st.session_state.results = None
if 'config' not in st.session_state:
    st.session_state.config = {
        'elastic_modulus_mode': 'multiplicative',
        'elastic_modulus_sigma': 0.4,
        'lambda_min': 0.35,
        'lambda_max': 1.20,
        'inference_steps': 8
    }
if 'audit_log' not in st.session_state:
    st.session_state.audit_log = deque(maxlen=100)


# ============================================================================
# Header
# ============================================================================

st.title("Deterministic Exclusion Demo")
st.markdown("Material-Field Engine GUI")
st.markdown("---")


# ============================================================================
# Sidebar: Configuration Panel
# ============================================================================

st.sidebar.header("Configuration")

# Elastic Modulus Mode
mode = st.sidebar.selectbox(
    "Elastic Modulus Mode",
    options=['cosine', 'multiplicative', 'rbf'],
    index=1,  # Default: multiplicative
    help="Cosine: direction only | Multiplicative: angle×proximity | RBF: proximity only"
)

# Sigma parameter
sigma = st.sidebar.slider(
    "Sigma (σ) - Field Extent",
    min_value=0.2,
    max_value=1.0,
    value=0.4,
    step=0.05,
    help="Lower = tighter binding, higher = looser binding"
)

st.sidebar.markdown(f"**Current: σ={sigma:.2f}**")

# Lambda max
lambda_max = st.sidebar.slider(
    "Lambda Max (λ_max) - Max Pressure",
    min_value=0.5,
    max_value=2.0,
    value=1.2,
    step=0.1,
    help="Higher = stricter exclusion"
)

st.sidebar.markdown("---")

# Run button
if st.sidebar.button("Run Deterministic Exclusion Demo", type="primary"):
    with st.spinner("Running inference..."):
        # Update config
        st.session_state.config['elastic_modulus_mode'] = mode
        st.session_state.config['elastic_modulus_sigma'] = sigma
        st.session_state.config['lambda_max'] = lambda_max

        # Run test
        results = run_deterministic_exclusion_demo(
            elastic_modulus_mode=mode,
            sigma=sigma,
            print_banner=False,
        )

        st.session_state.results = results

        # Log event
        st.session_state.audit_log.append({
            'timestamp': time.time(),
            'operation': 'run_inference',
            'mode': mode,
            'sigma': sigma,
            'hash': results['hash']
        })

        st.success("Inference complete.")


# ============================================================================
# Main Content Area
# ============================================================================

if st.session_state.results is None:
    st.info("Configure parameters and run the demo.")
    st.stop()

results = st.session_state.results


# ============================================================================
# Row 1: Deterministic Audit Trail
# ============================================================================

st.header("Deterministic Audit Trail")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("SHA-256")
    st.markdown(f'<div class="hash-display">{results["hash"]}</div>', unsafe_allow_html=True)

with col2:
    st.metric("Total Excluded", results["excluded"])


# ============================================================================
# Row 2: Abstention Indicator
# ============================================================================

st.header("Outcome Verification")

if results.get("winner_index") is None:
    st.error("Abstained")
else:
    st.success(f"Winner index: {results['winner_index']}")

st.write("Expected winner: Candidate 0")
st.write("Expected excluded count: 3")


# ============================================================================
# Row 3: Quench View (Phase Transition Visualization)
# ============================================================================

st.header("Phase Log")

# Create phase transition table
phase_log = results['phase_log']

# Convert to DataFrame for display
import pandas as pd
df = pd.DataFrame(phase_log)

# Styled table
st.dataframe(
    df[['step', 'phase', 'pressure', 'survivors', 'excluded']],
    use_container_width=True,
    hide_index=True
)

# Pressure ramp visualization
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=[entry['step'] for entry in phase_log],
    y=[entry['pressure'] for entry in phase_log],
    mode='lines+markers',
    name='Constraint Pressure λ(t)',
    line=dict(color='cyan', width=3),
    marker=dict(size=8)
))

# Add phase transition markers
nucleation_steps = [e['step'] for e in phase_log if e['phase'] == 'NUCLEATION']
quenching_steps = [e['step'] for e in phase_log if e['phase'] == 'QUENCHING']
crystallization_steps = [e['step'] for e in phase_log if e['phase'] == 'CRYSTALLIZATION']

if nucleation_steps:
    fig.add_trace(go.Scatter(
        x=nucleation_steps,
        y=[phase_log[s]['pressure'] for s in nucleation_steps],
        mode='markers',
        name='Nucleation',
        marker=dict(color='blue', size=12, symbol='diamond')
    ))

if quenching_steps:
    fig.add_trace(go.Scatter(
        x=quenching_steps,
        y=[phase_log[s]['pressure'] for s in quenching_steps],
        mode='markers',
        name='Quenching',
        marker=dict(color='yellow', size=12, symbol='square')
    ))

if crystallization_steps:
    fig.add_trace(go.Scatter(
        x=crystallization_steps,
        y=[phase_log[s]['pressure'] for s in crystallization_steps],
        mode='markers',
        name='Crystallization',
        marker=dict(color='green', size=12, symbol='star')
    ))

fig.update_layout(
    title="Constraint Pressure Ramp λ(t)",
    xaxis_title="Inference Step",
    yaxis_title="Pressure λ(t)",
    template="plotly_dark",
    height=400
)

st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# Row 4: Survivor Count Over Time
# ============================================================================

st.header("Survivor Count")

fig2 = go.Figure()

fig2.add_trace(go.Bar(
    x=[entry['step'] for entry in phase_log],
    y=[entry['survivors'] for entry in phase_log],
    name='Survivors',
    marker=dict(color='green')
))

fig2.add_trace(go.Bar(
    x=[entry['step'] for entry in phase_log],
    y=[entry['excluded'] for entry in phase_log],
    name='Excluded (this step)',
    marker=dict(color='red')
))

fig2.update_layout(
    title="Candidate Exclusion Dynamics",
    xaxis_title="Inference Step",
    yaxis_title="Count",
    barmode='group',
    template="plotly_dark",
    height=400
)

st.plotly_chart(fig2, use_container_width=True)


# ============================================================================
# Row 5: Audit Log (Recent Operations)
# ============================================================================

st.header("Run Log")

if st.session_state.audit_log:
    log_df = pd.DataFrame(st.session_state.audit_log)
    log_df['timestamp'] = pd.to_datetime(log_df['timestamp'], unit='s')

    st.dataframe(
        log_df[['timestamp', 'operation', 'mode', 'sigma', 'hash']],
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("No operations logged yet.")


# ============================================================================
# Footer
# ============================================================================

st.markdown("---")
st.caption("Development GUI")
