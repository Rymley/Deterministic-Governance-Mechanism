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

# ============================================================================
# Main Content Area
# ============================================================================

# Create tabs
tab1, tab2 = st.tabs(["Mechanism Demo", "LLM Guardrail"])

# ----------------------------------------------------------------------------
# TAB 1: Mechanism Demo (Original)
# ----------------------------------------------------------------------------
with tab1:
    if st.session_state.results is None:
        st.info("Configure parameters in the sidebar and click 'Run Deterministic Exclusion Demo'.")
    else:
        results = st.session_state.results
        
        # Row 1: Deterministic Audit Trail
        st.header("Deterministic Audit Trail")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("SHA-256")
            st.markdown(f'<div class="hash-display">{results["hash"]}</div>', unsafe_allow_html=True)
        with col2:
            st.metric("Total Excluded", results["excluded"])
            
        # Row 2: Abstention Indicator
        st.header("Outcome Verification")
        if results.get("winner_index") is None:
            st.error("Abstained")
        else:
            st.success(f"Winner index: {results['winner_index']}")
        st.caption("Expected winner: Candidate 0 | Expected excluded: 3")

        # Row 3: Phase Log
        st.header("Phase Log")
        phase_log = results['phase_log']
        import pandas as pd
        df = pd.DataFrame(phase_log)
        st.dataframe(df[['step', 'phase', 'pressure', 'survivors', 'excluded']], use_container_width=True, hide_index=True)

        # Visualization
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[e['step'] for e in phase_log],
            y=[e['pressure'] for e in phase_log],
            mode='lines+markers',
            name='Pressure λ(t)',
            line=dict(color='cyan', width=3)
        ))
        st.plotly_chart(fig, use_container_width=True)
        
        # Row 5: Audit Log
        st.header("Run Log")
        if st.session_state.audit_log:
            log_df = pd.DataFrame(st.session_state.audit_log)
            log_df['timestamp'] = pd.to_datetime(log_df['timestamp'], unit='s')
            st.dataframe(log_df[['timestamp', 'operation', 'mode', 'sigma', 'hash']], use_container_width=True, hide_index=True)


# ----------------------------------------------------------------------------
# TAB 2: LLM Guardrail Playground (New)
# ----------------------------------------------------------------------------
with tab2:
    st.header("Deterministic LLM Filter")
    st.markdown("""
    A model-agnostic post-processor that evaluates candidate outputs against a verified substrate.
    The mechanism deterministically accepts, rejects, or abstains based on explicit constraints.
    """)
    
    col_input1, col_input2 = st.columns(2)
    
    with col_input1:
        st.subheader("1. Verified Substrate")
        st.markdown("Approved facts (Ground Truth). One per line.")
        substrate_input = st.text_area(
            "Substrate",
            value="The sky is blue\nWater is wet\nParis is capital of France",
            height=200,
            key="llm_substrate"
        )
    
    with col_input2:
        st.subheader("2. LLM Candidates")
        st.markdown("Generated responses (including hallucinations). One per line.")
        candidates_input = st.text_area(
            "Candidates",
            value="The sky is blue\nThe sky is green\nThe sky is made of cheese",
            height=200,
            key="llm_candidates"
        )
        
    if st.button("Run Guardrail Filter", type="primary"):
        from llm_adapter import DeterministicGuardrail, MockEmbedder
        
        # Parse inputs
        substrate_list = [line.strip() for line in substrate_input.split('\n') if line.strip()]
        candidate_list = [line.strip() for line in candidates_input.split('\n') if line.strip()]
        
        if not substrate_list or not candidate_list:
            st.error("Please provide both substrate and candidates.")
        else:
            with st.spinner("Projecting to material field..."):
                # Initialize Guardrail
                # Use current sidebar config
                guard = DeterministicGuardrail(
                    substrate_texts=substrate_list,
                    config_preset='balanced'  # Default, but could use sidebar params if we passed them
                )
                
                # We want to INSPECT, not just filter
                inspection = guard.inspect(candidate_list)
                result_text = inspection['selected_text']
                metrics = inspection['metrics']
                
                # Display Result
                st.markdown("### Result")
                if result_text:
                    st.success(f"**Selected:** {result_text}")
                else:
                    st.warning("**Abstained**: No candidates met the yield strength requirements.")
                    
                # Visualization of the Field
                st.markdown("### Material Field Projection")
                
                # Get embeddings for visualization
                embedder = MockEmbedder()
                sub_vecs = [embedder.embed(t) for t in substrate_list]
                cand_vecs = [embedder.embed(t) for t in candidate_list]
                
                # Plot
                fig_map = go.Figure()
                
                # Plot Substrate (Green Squares)
                fig_map.add_trace(go.Scatter(
                    x=[v[0] for v in sub_vecs],
                    y=[v[1] for v in sub_vecs],
                    mode='markers',
                    name='Substrate (Facts)',
                    text=substrate_list,
                    marker=dict(symbol='square', size=12, color='green')
                ))
                
                # Plot Candidates (Red Circles)
                # Differentiate selected vs excluded
                selected_idx = metrics['final_output'].candidate_index if metrics['final_output'] else -1
                
                colors = ['gold' if i == selected_idx else 'red' for i in range(len(cand_vecs))]
                sizes = [15 if i == selected_idx else 10 for i in range(len(cand_vecs))]
                
                fig_map.add_trace(go.Scatter(
                    x=[v[0] for v in cand_vecs],
                    y=[v[1] for v in cand_vecs],
                    mode='markers+text',
                    name='Candidates',
                    text=candidate_list,
                    textposition='top center',
                    marker=dict(symbol='circle', size=sizes, color=colors)
                ))
                
                fig_map.update_layout(
                    title="Semantic Material Field (2D Mock Projection)",
                    xaxis_title="Dimension X",
                    yaxis_title="Dimension Y",
                    xaxis=dict(range=[0, 1]),
                    yaxis=dict(range=[0, 1]),
                    template="plotly_dark",
                    height=500
                )
                
                st.plotly_chart(fig_map, use_container_width=True)
                
                # Metrics
                st.json({
                    "survived": result_text is not None,
                    "total_excluded": metrics['total_excluded'],
                    "falsification_pressure": f"{metrics['phase_log'][-1]['pressure']:.2f}λ"
                })

# ============================================================================
# Footer
# ============================================================================

st.markdown("---")
st.caption("Development GUI | Deterministic Governance Mechanism")
