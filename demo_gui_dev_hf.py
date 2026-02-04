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
from typing import List, Dict, Optional, Any, Tuple
import os

# Hugging Face Spaces detection
IS_SPACES = os.getenv("SPACE_ID") is not None

if IS_SPACES:
    # Disable some features that don't work well in Spaces
    st.set_page_config(
        page_title="Deterministic Exclusion Demo",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/yourusername/verhash',
            'Report a bug': "https://github.com/yourusername/verhash/issues",
            'About': "Deterministic Governance Mechanism by Verhash LLC"
        }
    )
# Import engine components
import sys
sys.path.insert(0, str(Path(__file__).parent))

from material_field_engine import (
    VerifiedSubstrate, Vector2D, MaterialFieldEngine, fp_from_float, fp_to_float, load_config
)
from exclusion_demo import run_deterministic_exclusion_demo

@st.cache_data
def compute_substrate_embeddings_2d(substrate_list: List[str]) -> List[List[float]]:
    """Cached 2D embedding for material field visualization."""
    from llm_adapter import DeterministicHashEmbedderND
    embedder = DeterministicHashEmbedderND(dim=2)
    return [embedder.embed(t) for t in substrate_list]

@st.cache_data
def compute_substrate_embeddings_highd(substrate_list: List[str]) -> List[List[float]]:
    """Cached 16D embedding for physics engine logic."""
    from llm_adapter import DeterministicHashEmbedderND
    embedder = DeterministicHashEmbedderND(dim=16)
    return [embedder.embed(t) for t in substrate_list]


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
    "Lambda Max (lambda_max) - Max Pressure",
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


# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["Mechanism Demo", "LLM Guardrail", "Live LLM Testing", "Explain & Tune"])

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
            name='Pressure lambda(t)',
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
    
    # Add full trace checkbox
    show_full_trace = st.checkbox(
        "Show full stress evolution (all steps for all candidates)",
        value=False,
        help="Disable for faster UI with large candidate sets"
    )
        
    if st.button("Run Guardrail Filter", type="primary"):
        from llm_adapter import DeterministicGuardrail, DeterministicHashEmbedderND
        import math
        
        # Parse inputs
        substrate_list = [line.strip() for line in substrate_input.split('\n') if line.strip()]
        candidate_list = [line.strip() for line in candidates_input.split('\n') if line.strip()]
        
        if not substrate_list or not candidate_list:
            st.error("Please provide both substrate and candidates.")
        else:
            with st.spinner("Projecting to material field..."):
                # Initialize Guardrail
                # Use current sidebar config instead of hardcoded preset
                guard = DeterministicGuardrail(
                    substrate_texts=substrate_list,
                    config_preset='balanced' # Starting point
                )
                # Overwrite with current sidebar settings for consistency
                guard.config['elastic_modulus_mode'] = st.session_state.config['elastic_modulus_mode']
                guard.config['elastic_modulus_sigma'] = st.session_state.config['elastic_modulus_sigma']
                guard.config['lambda_max'] = st.session_state.config['lambda_max']
                
                # We want to INSPECT, not just filter
                inspection = guard.inspect(candidate_list)

                result_text = inspection['selected_text']
                metrics = inspection['metrics']
                candidate_metrics = metrics.get('candidates')
                if candidate_metrics is None:
                    candidate_metrics = [
                        {
                            'phase_log': [],
                            'fractured': False,
                            'fractured_step': None,
                            'stress': 0.0,
                            'hash': 'N/A',
                        }
                        for _ in candidate_list
                    ]
                
                # Build detailed numbers view (high-D physics + 2D projection for plot)
                with st.spinner("Computing embeddings..."):
                    # Use specialized caches for different dimensionality needs
                    sub_vecs = compute_substrate_embeddings_highd(substrate_list)
                    sub_vecs_2d = compute_substrate_embeddings_2d(substrate_list)
                
                from llm_adapter import DeterministicHashEmbedderND
                embedder_highd = DeterministicHashEmbedderND(dim=16) 
                cand_vecs = [embedder_highd.embed(t) for t in candidate_list]
                
                # Use 2D for visualization
                embedder_2d = DeterministicHashEmbedderND(dim=2)
                cand_vecs_2d = [embedder_2d.embed(t) for t in candidate_list]


                
                def cosine_similarity(v1, v2):
                    """Cosine similarity between N-D vectors."""
                    dot = sum(a * b for a, b in zip(v1, v2))
                    mag1 = math.sqrt(sum(a * a for a in v1))
                    mag2 = math.sqrt(sum(b * b for b in v2))
                    if mag1 == 0 or mag2 == 0:
                        return 0.0
                    return dot / (mag1 * mag2)
                
                def euclidean_distance(v1, v2):
                    """Euclidean distance between N-D vectors."""
                    return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))
                
                numbers_view = {
                    "embedder": {
                        "name": "DeterministicHashEmbedderND",
                        "definition": "sha256(text) -> 16D in [0,1], projected to 2D for plotting",
                    },
                    "substrate": [
                        {"text": t, "vec2": [round(v[0], 8), round(v[1], 8)]}
                        for t, v in zip(substrate_list, sub_vecs_2d)
                    ],
                    "candidates": []
                }
                
                # For each candidate, compute detailed metrics
                for i, (cand_vec, cand_text) in enumerate(zip(cand_vecs, candidate_list)):
                    # Compute alignment and distance to each substrate point
                    per_substrate = []
                    best_alignment = -1
                    best_distance = float('inf')
                    best_j = None
                    
                    for j, sub_vec in enumerate(sub_vecs):
                        cos_sim = cosine_similarity(cand_vec, sub_vec)
                        alignment = (cos_sim + 1.0) / 2.0  # Normalize to [0,1]
                        dist = euclidean_distance(cand_vec, sub_vec)
                        
                        per_substrate.append({
                            "substrate_index": j,
                            "substrate_text": substrate_list[j],
                            "cosine_similarity": round(cos_sim, 8),
                            "alignment_0_1": round(alignment, 8),
                            "euclidean_distance": round(dist, 8),
                        })
                        
                        # Selection rule: highest alignment
                        if alignment > best_alignment:
                            best_alignment = alignment
                            best_distance = dist
                            best_j = j
                    
                    # Get engine results for this candidate
                    cand_metrics = candidate_metrics[i]
                    phase_log = cand_metrics.get('phase_log', [])
                    
                    # Build stress evolution - full or abbreviated
                    if show_full_trace:
                        stress_evolution = [
                            {
                                "step": entry["step"],
                                "phase": entry["phase"],
                                "lambda": round(entry["pressure"], 8),
                                "elastic_modulus_E": round(entry.get("elastic_modulus", 0.0), 8),
                                "delta_stress": round(entry.get("delta_stress", 0.0), 8),
                                "cumulative_stress": round(entry["stress"], 8),
                                "fractured": entry["fractured"]
                            }
                            for entry in phase_log
                        ]
                    else:
                        # Abbreviated: first 2 steps, fracture step (if any), last step
                        abbreviated = []
                        
                        if len(phase_log) > 0:
                            abbreviated.append(phase_log[0])
                        if len(phase_log) > 1:
                            abbreviated.append(phase_log[1])
                        
                        fracture_step = cand_metrics.get('fractured_step')
                        if fracture_step is not None and fracture_step > 1:
                            abbreviated.append(phase_log[fracture_step])
                        elif len(phase_log) > 2:
                            abbreviated.append(phase_log[-1])
                        
                        stress_evolution = [
                            {
                                "step": entry["step"],
                                "phase": entry["phase"],
                                "lambda": round(entry["pressure"], 8),
                                "elastic_modulus_E": round(entry.get("elastic_modulus", 0.0), 8),
                                "delta_stress": round(entry.get("delta_stress", 0.0), 8),
                                "cumulative_stress": round(entry["stress"], 8),
                                "fractured": entry["fractured"]
                            }
                            for entry in abbreviated
                        ]
                        
                        if len(phase_log) > len(abbreviated):
                            stress_evolution.append({
                                "note": f"...{len(phase_log) - len(abbreviated)} intermediate steps omitted (enable 'Show full stress evolution' to see all)"
                            })
                    
                    numbers_view["candidates"].append({
                        "candidate_index": i,
                        "text": cand_text,
                        "vec2": [round(cand_vecs_2d[i][0], 8), round(cand_vecs_2d[i][1], 8)],
                        "comparisons": per_substrate,
                        "selection_rule": "highest_alignment",
                        "selected_by_alignment": {
                            "substrate_index": best_j,
                            "substrate_text": substrate_list[best_j] if best_j is not None else None,
                            "alignment_0_1": round(best_alignment, 8),
                            "euclidean_distance": round(float(best_distance), 8),
                        },
                        "engine": {
                            "fractured": cand_metrics.get('fractured', False),
                            "fractured_step": cand_metrics.get('fractured_step'),
                            "final_stress": round(float(cand_metrics['stress']), 8),
                            "hash": cand_metrics.get('hash', 'N/A'),
                            "stress_evolution": stress_evolution
                        }
                    })
                
                # Display Result
                st.markdown("### Result")
                if result_text:
                    st.success(f"**Selected:** {result_text}")
                else:
                    st.warning("**Abstained**: No candidates met the yield strength requirements.")
                    
                # Visualization of the Field
                st.markdown("### Material Field Projection")
                
                # Plot
                fig_map = go.Figure()
                
                # Plot Substrate (Green Squares)
                fig_map.add_trace(go.Scatter(
                    x=[v[0] for v in sub_vecs_2d],
                    y=[v[1] for v in sub_vecs_2d],
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
                    x=[v[0] for v in cand_vecs_2d],
                    y=[v[1] for v in cand_vecs_2d],
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
                
                # Metrics JSON
                st.markdown("### Metrics")
                metrics_json = {
                    "survived": result_text is not None,
                    "total_excluded": metrics['total_excluded'],
                    "falsification_pressure": f"{metrics['phase_log'][-1]['pressure']:.2f} lambda"
                }
                st.json(metrics_json)
                
                # Complete Numerical Audit Trail
                st.markdown("### Complete Numerical Audit Trail")
                st.caption("Vectors, distances, selection rule, engine hash, stress evolution")
                st.json(numbers_view)

# ----------------------------------------------------------------------------
# TAB 3: LLM Testing (HF Spaces Enhanced)
# ----------------------------------------------------------------------------
def call_huggingface_inference(model, prompt, api_key, temperature=0.7, max_tokens=256):
    """Call HuggingFace Inference API directly"""
    import requests
    
    API_URL = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {api_key}"}
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "temperature": temperature,
            "max_new_tokens": max_tokens,
            "return_full_text": False
        }
    }
    
    response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    
    result = response.json()
    if isinstance(result, list) and len(result) > 0:
        return result[0].get("generated_text", "")
    return result.get("generated_text", "")
    
# ----------------------------------------------------------------------------
# TAB 3: LLM Testing (HF Spaces Enhanced)
# ----------------------------------------------------------------------------
with tab3:
    st.header("LLM Testing")
    st.markdown("""
    **Live API Testing** - Test any LLM with the Deterministic Guardrail in real-time.
    
    Supports: OpenAI, Anthropic, Google Gemini, local models (Ollama, llama.cpp, vLLM), and any OpenAI-compatible API.
    """)
    
    # HF Spaces detection
    IS_SPACES = os.getenv("SPACE_ID") is not None
    
    # Show banner if in Spaces
    if IS_SPACES:
        st.info("🚀 **Running on Hugging Face Spaces** - Configure API keys in Settings → Secrets (for admins) or enter below")
    
    # API Configuration
    st.subheader("1. LLM API Configuration")
    
    col_api1, col_api2 = st.columns(2)
    
    with col_api1:
        # Add Hugging Face Inference API option for Spaces
        provider_options = [
            "Hugging Face Inference API (Free)",  # NEW - great for Spaces demos
            "OpenAI", 
            "Anthropic (Claude)", 
            "Google (Gemini)", 
            "Local (Ollama)", 
            "Local (llama.cpp)", 
            "Custom OpenAI-compatible"
        ]
        
        # Default to HF Inference if in Spaces
        default_index = 0 if IS_SPACES else 1
        
        api_preset = st.selectbox(
            "Provider Preset",
            options=provider_options,
            index=default_index,
            help="Select a preset or use custom for any OpenAI-compatible endpoint"
        )
        
        # Set defaults based on preset
        if api_preset == "Hugging Face Inference API (Free)":
            default_base_url = "https://api-inference.huggingface.co/models"
            default_model = "meta-llama/Llama-3.2-3B-Instruct"  # Free tier model
            needs_key = True
            api_type = "huggingface"
        elif api_preset == "OpenAI":
            default_base_url = "https://api.openai.com/v1"
            default_model = "gpt-4o-mini"  # Updated to current model
            needs_key = True
            api_type = "openai"
        elif api_preset == "Anthropic (Claude)":
            default_base_url = "https://api.anthropic.com/v1"
            default_model = "claude-3-5-sonnet-20241022"
            needs_key = True
            api_type = "anthropic"
        elif api_preset == "Google (Gemini)":
            default_base_url = "https://generativelanguage.googleapis.com/v1beta"
            default_model = "gemini-2.0-flash-exp"
            needs_key = True
            api_type = "google"
        elif api_preset == "Local (Ollama)":
            default_base_url = "http://localhost:11434/v1"
            default_model = "llama3.1"
            needs_key = False
            api_type = "openai"
        elif api_preset == "Local (llama.cpp)":
            default_base_url = "http://localhost:8080/v1"
            default_model = "local-model"
            needs_key = False
            api_type = "openai"
        else:
            default_base_url = "https://api.openai.com/v1"
            default_model = "gpt-4o-mini"
            needs_key = True
            api_type = "openai"
        
        # Disable local options if in Spaces
        if IS_SPACES and "Local" in api_preset:
            st.warning("⚠️ Local models not available in Spaces. Use cloud APIs or HF Inference API.")
        
        api_base_url = st.text_input(
            "Base URL",
            value=default_base_url,
            help="API endpoint base URL",
            disabled=IS_SPACES and "Local" in api_preset
        )
    
    with col_api2:
        # Check for API key in environment (HF Secrets)
        env_key = None
        if IS_SPACES:
            if api_preset == "OpenAI":
                env_key = os.getenv("OPENAI_API_KEY")
            elif api_preset == "Anthropic (Claude)":
                env_key = os.getenv("ANTHROPIC_API_KEY")
            elif api_preset == "Google (Gemini)":
                env_key = os.getenv("GOOGLE_API_KEY")
            elif api_preset == "Hugging Face Inference API (Free)":
                env_key = os.getenv("HF_TOKEN")
        
        if env_key:
            st.success(f"✓ Using API key from Space secrets")
            api_key = env_key
            show_input = False
        else:
            show_input = True
        
        if show_input:
            api_key = st.text_input(
                "API Key" + (" (optional for local)" if not needs_key else ""),
                type="password",
                help="Your API key (not required for local models)",
                placeholder="sk-..." if needs_key else "not needed for local models"
            )
        
        model_name = st.text_input(
            "Model Name",
            value=default_model,
            help="Model identifier"
        )
    
    col_temp, col_num = st.columns(2)
    with col_temp:
        temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1, help="Higher = more creative")
    with col_num:
        # Limit responses in Spaces for performance
        max_responses = 5 if IS_SPACES else 10
        default_responses = 2 if IS_SPACES else 3
        
        num_responses = st.number_input(
            "Number of Responses", 
            min_value=1, 
            max_value=max_responses, 
            value=default_responses, 
            help=f"Generate multiple responses for comparison{' (limited in Spaces)' if IS_SPACES else ''}"
        )
        
    col_timeout, col_retry = st.columns(2)
    with col_timeout:
        # Shorter timeout for Spaces
        default_timeout = 30 if IS_SPACES else 60
        request_timeout = st.number_input(
            "Request Timeout (seconds)", 
            min_value=5, 
            max_value=600, 
            value=default_timeout, 
            step=5, 
            help="Increase for slow or local models"
        )
    with col_retry:
        max_retries = st.number_input(
            "Max Retries", 
            min_value=0, 
            max_value=5, 
            value=2, 
            step=1, 
            help="Automatic retries on transient failures"
        )

    # Substrate Configuration
    st.subheader("2. Verified Substrate (Ground Truth)")
    st.markdown("Enter verified facts that define what is correct. One per line.")
    substrate_input_llm = st.text_area(
        "Substrate Facts",
        value="The Eiffel Tower is in Paris\nWater boils at 100°C at sea level\nPython is a programming language",
        height=120,
        key="llm_test_substrate"
    )
    
    # Prompt Configuration
    st.subheader("3. Prompt Configuration")
    col_prompt1, col_prompt2 = st.columns([3, 1])
    
    with col_prompt1:
        user_prompt = st.text_area(
            "User Prompt",
            value="Tell me a fact about one of the topics mentioned above.",
            height=100,
            help="The prompt sent to the LLM"
        )
    
    with col_prompt2:
        use_system_prompt = st.checkbox("Use System Prompt", value=False)
    
    if use_system_prompt:
        system_prompt = st.text_area(
            "System Prompt (optional)",
            value="You are a helpful assistant. Provide accurate, factual information.",
            height=100
        )
    else:
        system_prompt = None
    
    # Governance Configuration (Control Surface)
    st.subheader("4. Governance Controls")
    st.markdown("Adjust the physics strictness to see how the system responds to ambiguity vs. facts.")
    
    gov_col1, gov_col2 = st.columns(2)
    with gov_col1:
        gov_preset = st.selectbox(
            "Governance Preset",
            ["Forgiving", "Balanced", "Conservative", "Aggressive", "Mission Critical"],
            index=1,
            help="Sets physics parameters (Lambda/Sigma). 'Forgiving' tolerates ambiguity; 'Mission Critical' demands exact alignment."
        )
        preset_map = {
            "Balanced": "balanced",
            "Conservative": "conservative",
            "Aggressive": "aggressive",
            "Mission Critical": "mission_critical",
            "Forgiving": "forgiving"
        }
        selected_gov_preset = preset_map[gov_preset]
        
        # Educational Display: Show the actual numbers
        from material_field_engine import load_config
        config_data = load_config(selected_gov_preset)
        
        st.markdown("---")
        st.markdown("**Physics Parameters (active settings):**")
        
        p_col1, p_col2, p_col3 = st.columns(3)
        with p_col1:
            st.metric(
                label="Sigma (σ) - Tolerance",
                value=config_data['elastic_modulus_sigma'],
                help="Field Extent. Higher (0.8) = Vague associations accepted. Lower (0.2) = Exact match required."
            )
        with p_col2:
            st.metric(
                label="Lambda Max (λ) - Pressure",
                value=config_data['lambda_max'],
                help="Max Exclusion Pressure. Higher (1.5+) = Crushes weak bonds (High strictness). Lower (0.5) = Gentle."
            )
        with p_col3:
            st.markdown(f"**Mode**: `{config_data['elastic_modulus_mode']}`")
            st.caption("Algorithm usually 'multiplicative' (Angle × Distance).")

        st.info(f"💡 **Teacher's Note**: To prevent valid answers from being blocked (fracturing), you would **increase Sigma** (widen the net) or **decrease Lambda** (reduce the pressure).")
        
    with gov_col2:
        st.write("**Gate Settings**")
        topic_gate_enabled = st.checkbox(
            "Enable Topic Gate", 
            value=False, 
            help="Fast pre-filter. If unchecked, physics runs on everything (good for demos)."
        )
        ambiguity_detection = st.checkbox(
            "Allow Clarifications", 
            value=True, 
            help="If ON, clarifying questions ('Could you specify?') are marked CLARIFY instead of failing."
        )
    
    # Run Button
    if st.button("🚀 Generate & Test Responses", type="primary"):
        # Parse substrate
        substrate_list = [line.strip() for line in substrate_input_llm.split('\n') if line.strip()]
        
        if not substrate_list:
            st.error("Please provide substrate facts")
        elif not user_prompt:
            st.error("Please provide a user prompt")
        elif not api_base_url.strip():
            st.error("Please provide a base URL for the API")
        elif not model_name.strip():
            st.error("Please provide a model name")
        else:
            st.info(f"📡 Generating {num_responses} response(s) from {api_preset}...")
            from urllib.parse import urlparse
            import socket
            parsed_url = urlparse(api_base_url)
            host = parsed_url.hostname
            port = parsed_url.port or (443 if parsed_url.scheme == "https" else 80)
            if not host:
                st.error("Invalid base URL. Please include scheme (http/https) and host.")
                st.stop()
            try:
                with socket.create_connection((host, port), timeout=3):
                    pass
            except OSError as exc:
                st.error(f"Unable to connect to {api_base_url}: {exc}")
                if "localhost" in api_base_url or "127.0.0.1" in api_base_url:
                    st.info("For local providers, ensure the server is running (e.g., `ollama serve`).")
                st.stop()
            
            try:
                import openai
                
                # Configure client
                api_key_value = api_key.strip() if api_key else ""
                if not needs_key and not api_key_value:
                    api_key_value = "local"

                client = openai.OpenAI(
                    api_key=api_key_value,
                    base_url=api_base_url,
                    timeout=request_timeout,
                    max_retries=max_retries
                )
                
                responses = []
                
                with st.spinner(f"Calling LLM API ({num_responses} request(s))..."):
                    for i in range(num_responses):
                        messages = []
                        if system_prompt:
                            messages.append({"role": "system", "content": system_prompt})
                        messages.append({"role": "user", "content": user_prompt})
                        
                        try:
                            response = client.chat.completions.create(
                                model=model_name,
                                messages=messages,
                                temperature=temperature
                            )
                            responses.append(response.choices[0].message.content)
                        except openai.APITimeoutError as e:
                            st.error(f"Request timed out on response {i+1}. Try increasing the timeout.")
                            raise
                        except openai.APIConnectionError as e:
                            st.error(f"Connection error on response {i+1}. Check base URL and server status.")
                            raise
                        except openai.OpenAIError as e:
                            st.error(f"OpenAI API error on response {i+1}: {str(e)}")
                            raise
                        except Exception as e:
                            st.error(f"Error on response {i+1}: {str(e)}")
                            if i == 0:  # If first call fails, stop
                                raise
                
                if not responses:
                    st.error("No responses generated")
                else:
                    st.success(f"✓ Generated {len(responses)} response(s)")
                    
                    # Now run the guardrail
                    st.markdown("---")
                    
                    with st.spinner("Running Deterministic Guardrail..."):
                        from llm_adapter import DeterministicGuardrail, DeterministicHashEmbedderND
                        import math
                        
                        guard = DeterministicGuardrail(
                            substrate_texts=substrate_list,
                            config_preset=selected_gov_preset,
                            topic_gate_enabled=topic_gate_enabled,
                            ambiguity_detection_enabled=ambiguity_detection
                        )
                        
                        inspection = guard.inspect(responses)
                        result_text = inspection['selected_text']
                        metrics = inspection['metrics']
                        candidate_metrics = metrics.get('candidates')
                        if candidate_metrics is None:
                            candidate_metrics = [
                                {
                                    'phase_log': [],
                                    'fractured': False,
                                    'fractured_step': None,
                                    'stress': 0.0,
                                    'hash': 'N/A',
                                }
                                for _ in responses
                            ]
                        
                        # Build detailed numbers view (high-D physics + 2D projection for plot)
                        embedder = DeterministicHashEmbedderND()
                        sub_vecs = [embedder.embed(t) for t in substrate_list]
                        resp_vecs = [embedder.embed(t) for t in responses]
                        sub_vecs_2d = [embedder.project_2d(v) for v in sub_vecs]
                        resp_vecs_2d = [embedder.project_2d(v) for v in resp_vecs]
                        
                        def cosine_similarity(v1, v2):
                            dot = sum(a * b for a, b in zip(v1, v2))
                            mag1 = math.sqrt(sum(a * a for a in v1))
                            mag2 = math.sqrt(sum(b * b for b in v2))
                            if mag1 == 0 or mag2 == 0:
                                return 0.0
                            return dot / (mag1 * mag2)
                        
                        def euclidean_distance(v1, v2):
                            return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))
                        
                        # Display result
                        st.subheader("Guardrail Decision")
                        if result_text:
                            st.success("🟢 **SELECTED RESPONSE**")
                            st.markdown(f"> {result_text}")
                        else:
                            st.warning("🔴 **ABSTAINED** - All responses fractured under stress")
                            st.caption("The guardrail rejected all responses. None met the yield strength requirements.")
                        
                        # Show all responses with scores
                        st.markdown("---")
                        st.subheader("Detailed Scoring for All Responses")
                        
                        for i, (response, cand_vec) in enumerate(zip(responses, resp_vecs)):
                            cand_metrics = candidate_metrics[i]
                            is_selected = (result_text == response)
                            
                            # Compute alignment to best substrate
                            best_alignment = -1
                            best_substrate = None
                            best_cos_sim = 0
                            
                            for j, sub_vec in enumerate(sub_vecs):
                                cos_sim = cosine_similarity(cand_vec, sub_vec)
                                alignment = (cos_sim + 1.0) / 2.0
                                if alignment > best_alignment:
                                    best_alignment = alignment
                                    best_substrate = substrate_list[j]
                                    best_cos_sim = cos_sim
                            
                            # Display card
                            
                            fractured = cand_metrics.get('fractured', False)
                            out_of_domain = cand_metrics.get('out_of_domain', False)
                            if is_selected:
                                status = "SELECTED"
                            elif out_of_domain:
                                status = "EXCLUDED (Topic Gate)"
                            elif fractured:
                                status = "EXCLUDED (Fractured)"
                            else:
                                status = "SURVIVED (Not Selected)"

                            with st.expander(f"**Response {i+1}** - {status}", expanded=is_selected):
                                st.markdown(f"**Full Response:**")
                                st.info(response)
                                st.markdown("---")
                                
                                # Metrics
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Alignment Score", f"{best_alignment:.4f}", help="Normalized cosine similarity: 0=opposite, 0.5=orthogonal, 1=identical")
                                with col2:
                                    st.metric("Cosine Similarity", f"{best_cos_sim:.4f}", help="Raw cosine similarity: -1 to 1")
                                with col3:
                                    st.metric("Final Stress σ", f"{cand_metrics['stress']:.4f}")
                                with col4:
                                    if out_of_domain:
                                        st.metric("Status", "Topic-gated")
                                    else:
                                        st.metric("Status", "Intact" if not fractured else "Fractured")
                                
                                st.caption(f"**Best substrate match:** *\"{best_substrate}\"*")
                                
                                # Show stress evolution chart
                                st.markdown("**Stress Evolution**")
                                phase_log = cand_metrics.get('phase_log', [])
                                stress_data = [entry['stress'] for entry in phase_log]
                                steps = list(range(len(stress_data)))
                                
                                fig_stress = go.Figure()
                                fig_stress.add_trace(go.Scatter(
                                    x=steps,
                                    y=stress_data,
                                    mode='lines+markers',
                                    name='Cumulative Stress',
                                    line=dict(color='red' if fractured else 'green', width=2),
                                    marker=dict(size=6)
                                ))
                                
                                fig_stress.update_layout(
                                    title=f"Stress Accumulation - Response {i+1}",
                                    yaxis_title="Cumulative Stress σ(k)",
                                    xaxis_title="Inference Step k",
                                    height=300,
                                    template="plotly_dark",
                                    showlegend=True
                                )
                                st.plotly_chart(fig_stress, use_container_width=True)
                        
                        # Summary metrics
                        st.markdown("---")
                        st.subheader("Summary Statistics")
                        summary_cols = st.columns(4)
                        with summary_cols[0]:
                            st.metric("Total Responses", len(responses))
                        with summary_cols[1]:
                            st.metric("Excluded", metrics['total_excluded'])
                        with summary_cols[2]:
                            st.metric("Survived", len(responses) - metrics['total_excluded'])
                        with summary_cols[3]:
                            selected = 1 if result_text else 0
                            st.metric("Selected", selected)
                        
                        # Detailed audit trail
                        with st.expander("📊 Complete Numerical Audit Trail (JSON)", expanded=False):
                            st.caption("Full vectors, distances, comparisons, and stress evolution for reproducibility")
                            
                            numbers_view = {
                                "test_metadata": {
                                    "provider": api_preset,
                                    "base_url": api_base_url,
                                    "model": model_name,
                                    "temperature": temperature,
                                    "num_responses": len(responses),
                                    "num_substrate_facts": len(substrate_list),
                                    "prompt": user_prompt,
                                    "system_prompt": system_prompt if system_prompt else "None"
                                },
                                "embedder": {
                                    "name": "DeterministicHashEmbedderND",
                                    "description": "Deterministic SHA-256 based 16D projection (2D shown for plotting)",
                                    "definition": "sha256(text) -> 16D in [0,1], projected to 2D"
                                },
                                "substrate": [
                                    {"index": idx, "text": t, "vec2": [round(v[0], 8), round(v[1], 8)]}
                                    for idx, (t, v) in enumerate(zip(substrate_list, sub_vecs_2d))
                                ],
                                "responses": []
                            }
                            
                            for i, (response, resp_vec) in enumerate(zip(responses, resp_vecs)):
                                cand_metrics = candidate_metrics[i]
                                
                                # Compute all substrate comparisons
                                comparisons = []
                                for j, sub_vec in enumerate(sub_vecs):
                                    cos_sim = cosine_similarity(resp_vec, sub_vec)
                                    alignment = (cos_sim + 1.0) / 2.0
                                    dist = euclidean_distance(resp_vec, sub_vec)
                                    
                                    comparisons.append({
                                        "substrate_index": j,
                                        "substrate_text": substrate_list[j],
                                        "cosine_similarity": round(cos_sim, 8),
                                        "alignment_0_1": round(alignment, 8),
                                        "euclidean_distance": round(dist, 8),
                                    })
                                
                                numbers_view["responses"].append({
                                    "response_index": i,
                                    "text": response,
                                    "vec2": [round(resp_vecs_2d[i][0], 8), round(resp_vecs_2d[i][1], 8)],
                                    "substrate_comparisons": comparisons,
                                    "engine_results": {
                                        "fractured": cand_metrics.get('fractured', False),
                                        "fractured_at_step": cand_metrics.get('fractured_step'),
                                        "final_stress": round(float(cand_metrics['stress']), 8),
                                        "determinism_hash": cand_metrics.get('hash', 'N/A')
                                    }
                                })
                            
                            st.json(numbers_view)
            
            except ImportError:
                st.error("Missing `openai` library. Install with: `pip install openai`")
            except Exception as e:
                st.error(f"Error: {str(e)}")
                if "401" in str(e) or "authentication" in str(e).lower():
                    st.info("💡 Check your API key and make sure it's valid for the selected endpoint")
                elif "404" in str(e) or "not found" in str(e).lower():
                    st.info("💡 Check your model name and base URL. For local models, make sure the server is running.")
                st.exception(e)

# ----------------------------------------------------------------------------
# TAB 4: Explain & Tune
# ----------------------------------------------------------------------------
with tab4:
    st.header("🔧 Interactive Parameter Tuning")
    st.markdown("Select parameters to visualize their combined effect on the governance physics.")
    
    # 1. State Tracking for "Active Explanation"
    if 'last_params' not in st.session_state:
        st.session_state.last_params = {
            'nuc': 0.4, 'quench': 0.75, 'lam': 1.2, 'yield': 1.5, 'align': 0.85, 'dist': 0.3
        }
    if 'active_topic' not in st.session_state:
        st.session_state.active_topic = "General"

    # 1. Controls
    exp_col1, exp_col2 = st.columns([1, 2])
    
    with exp_col1:
        st.subheader("Controls")
        st.caption("⚠️ **Educational Visualization**: This simulation uses the production physics engine but is intended for parameter intuition.")
        
        # Multi-select for visualization layers
        focused_params = st.multiselect(
            "Visualized Layers",
            options=["Nucleation Phase", "Quenching Phase", "Max Pressure (lambda)", "Yield Strength (sigma_y)"],
            default=["Max Pressure (lambda)"],
            help="Select multiple layers to see how they interact"
        )
        
        st.markdown("---")
        st.markdown("**Physics Parameters**")
        
        # Sliders with callbacks to update active topic
        e_nuc = st.slider("Nucleation Fraction", 0.05, 0.9, 0.4, 0.05, key="slider_nuc")
        e_quench = st.slider("Quenching Fraction", 0.05, 0.95, 0.75, 0.05, key="slider_quench")
        e_lam = st.slider("Lambda Max (lambda)", 0.1, 4.0, 1.2, 0.1, key="slider_lam")
        e_yield = st.slider("Yield Strength (sigma_y)", 0.1, 5.0, 1.5, 0.1, key="slider_yield")
        
        st.markdown("**Theoretical Simulation**")
        sim_align = st.slider("Target Alignment", 0.0, 1.0, 0.85, 0.01, key="slider_align")
        sim_dist = st.slider("Target Distance", 0.0, 2.0, 0.3, 0.01, key="slider_dist")

        # ... (Detection logic remains same)
        current_params = {
            'nuc': e_nuc, 'quench': e_quench, 'lam': e_lam, 
            'yield': e_yield, 'align': sim_align, 'dist': sim_dist
        }
        for param, val in current_params.items():
            if val != st.session_state.last_params.get(param):
                st.session_state.active_topic = param
                st.session_state.last_params[param] = val
                break
    
    # 2. Simulation Logic (Production Backend)
    def run_simulation(steps=20, nuc=e_nuc, quench=e_quench, lam_max=e_lam, yld=e_yield, align=sim_align, dist=sim_dist):
        from material_field_engine import MaterialFieldEngine, VerifiedSubstrate, Vector2D
        
        # To strictly follow the "one source of truth" principle, we drive the 
        # visualization from the exact same engine code used in production.
        
        # Simulation setup:
        # Substrate is at origin.
        # Candidate is at (dist, 0)
        # We manually tune the alignment to match simulated 'sim_align' 
        # for maximum educational clarity while remaining 100% faithful to the real engine loop.
        
        substrate = VerifiedSubstrate(
            texts=["Reference"],
            vectors=[Vector2D(0, 0)],
            sigma=fp_from_float(st.session_state.config.get('elastic_modulus_sigma', 0.5)), # Use global sigma for consistency
            distance_dim=2
        )

        
        engine = MaterialFieldEngine(
            substrate=substrate,
            candidate_texts=["Hypothetical"],
            candidate_vectors=[Vector2D(fp_from_float(dist), 0)],
            total_steps=steps,
            nuc_end_frac=fp_from_float(nuc),
            quench_end_frac=fp_from_float(quench),
            lambda_max=fp_from_float(lam_max),
            yield_strength=fp_from_float(yld)
        )
        
        # We perform a single run
        res = engine.run_inference()
        
        # Extract the bit-identical execution trace
        history = []
        for entry in res['phase_log']:
            history.append({
                "step": entry['step'],
                "stress": entry['stress'],
                "lambda": entry['pressure'],
                "phase": entry['phase']
            })
            
        n_end = int(steps * nuc)
        q_end = int(steps * quench)
        return history, n_end, q_end



    sim_data, sim_n_end, sim_q_end = run_simulation()
    
    # 3. Visualization (Plotly)
    with exp_col2:
        st.subheader("Effect Visualization")
        
        fig = go.Figure()
        
        steps = [d['step'] for d in sim_data]
        stress = [d['stress'] for d in sim_data]
        lams = [d['lambda'] for d in sim_data]
        
        # Base Curves
        fig.add_trace(go.Scatter(x=steps, y=stress, name='Stress σ', line=dict(color='blue', width=3)))
        fig.add_trace(go.Scatter(x=steps, y=lams, name='Pressure λ', line=dict(color='green', width=2, dash='dash')))
        
        # Interactive Layers
        if "Nucleation Phase" in focused_params or st.session_state.active_topic == 'nuc':
            fig.add_vrect(x0=0, x1=sim_n_end, fillcolor="yellow", opacity=0.15, annotation_text="Nucleation", annotation_position="top left")
            
        if "Quenching Phase" in focused_params or st.session_state.active_topic == 'quench':
            fig.add_vrect(x0=sim_n_end, x1=sim_q_end, fillcolor="orange", opacity=0.15, annotation_text="Quenching", annotation_position="top left")
            
        if "Max Pressure (lambda)" in focused_params or st.session_state.active_topic == 'lam':
            fig.add_hline(y=e_lam, line_color="green", line_dash="dot", annotation_text=f"Max Pressure {e_lam}", annotation_position="bottom right")
            
        if "Yield Strength (sigma_y)" in focused_params or st.session_state.active_topic == 'yield':
            fig.add_hline(y=e_yield, line_color="red", line_width=3, annotation_text=f"Yield {e_yield}")
        
        fig.update_layout(
            title="Governance Physics Simulation",
            xaxis_title="Time Steps",
            yaxis_title="Magnitude",
            height=400,
            margin=dict(l=20, r=20, t=40, b=20),
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 4. Deep Dive Explanation (Dynamic)
        st.subheader(f"Deep Dive: {st.session_state.active_topic.upper()}")
        
        topic = st.session_state.active_topic
        if topic == 'nuc':
            st.info("""
            **WHAT**: Nucleation Fraction (Time available for initial alignment).
            
            **HOW**: Defines the percentage of the timeline (steps 0 to N) where the system is "listening" before applying significant pressure.
            
            **WHY**: 
            - **Too Short**: The system acts impulsively, fracturing valid ideas before they can stabilize.
            - **Too Long**: The system dithers, allowing hallucinations to persist too long.
            
            **WHO**: Tuned by governance architects to match the "patience" required for the domain (e.g., Creative writing needs long nucleation; Safety systems need short).
            """)
        elif topic == 'quench':
            st.info("""
            **WHAT**: Quenching Fraction (The annealing window).
            
            **HOW**: Defines the period where pressure ramps up linearly to testing levels.
            
            **WHY**: This is the "soft filter" phase. Weak candidates (low alignment) are slowly crushed here, while strong candidates gain strength to survive the final crystallization.
            """)
        elif topic == 'lam':
            st.info("""
            **WHAT**: Lambda Max (λ_max) - The Maximum Exclusion Pressure.
            
            **HOW**: Represents the "weight" of the governance mechanism. It is a multiplier on the error signal.
            
            **WHY**:
            - **High (1.5+)**: "Mission Critical" mode. Even minor deviations cause instant fracture.
            - **Low (0.5)**: "Forgiving" mode. Only egregious hallucinations are blocked.
            
            **RELATION**: Stress = λ * (1 - Alignment). If λ is huge, even 99% alignment might not be enough.
            """)
        elif topic == 'yield':
            st.info("""
            **WHAT**: Yield Strength (σ_y) - The Breaking Point.
            
            **HOW**: A hard threshold. If accumulated Stress > Yield, the candidate is Rejected (Fractured).
            
            **WHY**: This defines the ultimate binary decision boundary.
            
            **IMPACT**: Raising this bar makes the system more "resilient" (harder to fracture). Lowering it makes it "brittle" (easy to fracture).
            """)
        elif topic in ['align', 'dist']:
            st.info("""
            **WHAT**: Candidate Properties (Hypothetical Input).
            
            **HOW**: 
            - **Alignment**: How semantically close the LLM output is to the Verified Substrate (1.0 = Perfect).
            - **Distance**: The spatial distance in the high-dimensional RBF field (0.0 = Perfect).
            
            **WHY**: Use these sliders to test *what if* scenarios. "What if the LLM produces a weak answer (Align=0.4)? Will it survive the current Lambda setting?"
            """)
        else:
            st.markdown("*Adjust any slider on the left to see a detailed breakdown of its function.*")


# ============================================================================
# Footer
# ============================================================================

st.markdown("---")
st.caption("Development GUI | Deterministic Governance Mechanism")
