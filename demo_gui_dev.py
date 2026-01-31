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


# Create tabs
tab1, tab2, tab3 = st.tabs(["Mechanism Demo", "LLM Guardrail", "Live LLM Testing"])

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
    
    # Add full trace checkbox
    show_full_trace = st.checkbox(
        "Show full stress evolution (all steps for all candidates)",
        value=False,
        help="Disable for faster UI with large candidate sets"
    )
        
    if st.button("Run Guardrail Filter", type="primary"):
        from llm_adapter import DeterministicGuardrail, MockEmbedder
        import math
        
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
                
                # Build detailed numbers view
                embedder = MockEmbedder()
                sub_vecs = [embedder.embed(t) for t in substrate_list]
                cand_vecs = [embedder.embed(t) for t in candidate_list]
                
                def cosine_similarity(v1, v2):
                    """Cosine similarity between 2D vectors."""
                    dot = v1[0] * v2[0] + v1[1] * v2[1]
                    mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
                    mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
                    if mag1 == 0 or mag2 == 0:
                        return 0.0
                    return dot / (mag1 * mag2)
                
                def euclidean_distance(v1, v2):
                    """Euclidean distance between 2D vectors."""
                    return math.sqrt((v1[0] - v2[0])**2 + (v1[1] - v2[1])**2)
                
                numbers_view = {
                    "embedder": {
                        "name": "MockEmbedder",
                        "definition": "sha256(text) -> first 8 bytes -> two uint32 scaled to [0,1]",
                    },
                    "substrate": [
                        {"text": t, "vec2": [round(v[0], 8), round(v[1], 8)]}
                        for t, v in zip(substrate_list, sub_vecs)
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
                    cand_metrics = metrics['candidates'][i]
                    phase_log = cand_metrics['phase_log']
                    
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
                        "vec2": [round(cand_vec[0], 8), round(cand_vec[1], 8)],
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
                
                # Metrics JSON
                st.markdown("### Metrics")
                metrics_json = {
                    "survived": result_text is not None,
                    "total_excluded": metrics['total_excluded'],
                    "falsification_pressure": f"{metrics['phase_log'][-1]['pressure']:.2f}λ"
                }
                st.json(metrics_json)
                
                # Complete Numerical Audit Trail
                st.markdown("### Complete Numerical Audit Trail")
                st.caption("Vectors, distances, selection rule, engine hash, stress evolution")
                st.json(numbers_view)

# ----------------------------------------------------------------------------
# TAB 3: Universal LLM Testing
# ----------------------------------------------------------------------------
with tab3:
    st.header("Universal LLM Testing")
    st.markdown("""
    **Live API Testing** - Test any LLM with the Deterministic Guardrail in real-time.
    
    Supports: OpenAI, Anthropic, Google Gemini, local models (Ollama, llama.cpp, vLLM), and any OpenAI-compatible API.
    """)
    
    # API Configuration
    st.subheader("1. LLM API Configuration")
    
    col_api1, col_api2 = st.columns(2)
    
    with col_api1:
        api_preset = st.selectbox(
            "Provider Preset",
            options=["OpenAI", "Anthropic (Claude)", "Google (Gemini)", "Local (Ollama)", "Local (llama.cpp)", "Custom OpenAI-compatible"],
            help="Select a preset or use custom for any OpenAI-compatible endpoint"
        )
        
        # Set defaults based on preset
        if api_preset == "OpenAI":
            default_base_url = "https://api.openai.com/v1"
            default_model = "gpt-4o-mini"
            needs_key = True
        elif api_preset == "Anthropic (Claude)":
            default_base_url = "https://api.anthropic.com/v1"
            default_model = "claude-3-5-sonnet-20241022"
            needs_key = True
        elif api_preset == "Google (Gemini)":
            default_base_url = "https://generativelanguage.googleapis.com/v1beta"
            default_model = "gemini-2.0-flash-exp"
            needs_key = True
        elif api_preset == "Local (Ollama)":
            default_base_url = "http://localhost:11434/v1"
            default_model = "llama3.1"
            needs_key = False
        elif api_preset == "Local (llama.cpp)":
            default_base_url = "http://localhost:8080/v1"
            default_model = "local-model"
            needs_key = False
        else:
            default_base_url = "https://api.openai.com/v1"
            default_model = "gpt-4o-mini"
            needs_key = True
        
        api_base_url = st.text_input(
            "Base URL",
            value=default_base_url,
            help="API endpoint base URL"
        )
    
    with col_api2:
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
        num_responses = st.number_input("Number of Responses", min_value=1, max_value=10, value=3, help="Generate multiple responses for comparison")
    
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
    
    # Run Button
    if st.button("🚀 Generate & Test Responses", type="primary"):
        # Parse substrate
        substrate_list = [line.strip() for line in substrate_input_llm.split('\n') if line.strip()]
        
        if not substrate_list:
            st.error("⚠️ Please provide substrate facts")
        elif not user_prompt:
            st.error("⚠️ Please provide a user prompt")
        else:
            st.info(f"📡 Generating {num_responses} response(s) from {api_preset}...")
            
            try:
                import openai
                
                # Configure client
                client = openai.OpenAI(
                    api_key=api_key if api_key else "not-needed",
                    base_url=api_base_url
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
                        from llm_adapter import DeterministicGuardrail, MockEmbedder
                        import math
                        
                        guard = DeterministicGuardrail(
                            substrate_texts=substrate_list,
                            config_preset='balanced'
                        )
                        
                        inspection = guard.inspect(responses)
                        result_text = inspection['selected_text']
                        metrics = inspection['metrics']
                        
                        # Build detailed numbers view
                        embedder = MockEmbedder()
                        sub_vecs = [embedder.embed(t) for t in substrate_list]
                        resp_vecs = [embedder.embed(t) for t in responses]
                        
                        def cosine_similarity(v1, v2):
                            dot = v1[0] * v2[0] + v1[1] * v2[1]
                            mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
                            mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
                            if mag1 == 0 or mag2 == 0:
                                return 0.0
                            return dot / (mag1 * mag2)
                        
                        def euclidean_distance(v1, v2):
                            return math.sqrt((v1[0] - v2[0])**2 + (v1[1] - v2[1])**2)
                        
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
                            cand_metrics = metrics['candidates'][i]
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
                            if is_selected:
                                status = "🟢 SELECTED"
                            elif fractured:
                                status = "🔴 EXCLUDED (Fractured)"
                            else:
                                status = "⚪ SURVIVED (Not Selected)"
                            
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
                                    st.metric("Status", "Intact ✓" if not fractured else "Fractured ✗")
                                
                                st.caption(f"**Best substrate match:** *\"{best_substrate}\"*")
                                
                                # Show stress evolution chart
                                st.markdown("**Stress Evolution**")
                                phase_log = cand_metrics['phase_log']
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
                                    "name": "MockEmbedder",
                                    "description": "Deterministic SHA-256 based 2D projection",
                                    "definition": "sha256(text.strip().lower()) -> first 8 bytes for X, next 8 for Y, normalized to [0,1]"
                                },
                                "substrate": [
                                    {"index": idx, "text": t, "vec2": [round(v[0], 8), round(v[1], 8)]}
                                    for idx, (t, v) in enumerate(zip(substrate_list, sub_vecs))
                                ],
                                "responses": []
                            }
                            
                            for i, (response, resp_vec) in enumerate(zip(responses, resp_vecs)):
                                cand_metrics = metrics['candidates'][i]
                                
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
                                    "vec2": [round(resp_vec[0], 8), round(resp_vec[1], 8)],
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

# ============================================================================
# Footer
# ============================================================================

st.markdown("---")
st.caption("Development GUI | Deterministic Governance Mechanism")
