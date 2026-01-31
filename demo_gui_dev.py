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
# TAB 3: Live LLM Testing (New)
# ----------------------------------------------------------------------------
with tab3:
    st.header("Live LLM Testing")
    st.markdown("""
    Test real LLM APIs and see how the Deterministic Guardrail scores their responses.
    Supports OpenAI, Anthropic, Google, and other providers.
    """)
    
    # API Configuration
    st.subheader("1. API Configuration")
    col_api1, col_api2 = st.columns(2)
    
    with col_api1:
        api_provider = st.selectbox(
            "LLM Provider",
            options=["OpenAI", "Anthropic", "Google (Gemini)", "Custom (OpenAI-compatible)"],
            help="Select your LLM provider"
        )
        
        api_key = st.text_input(
            "API Key",
            type="password",
            help="Your API key will not be stored",
            placeholder="sk-..."
        )
    
    with col_api2:
        if api_provider == "OpenAI":
            model_name = st.selectbox(
                "Model",
                options=["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
                index=1
            )
        elif api_provider == "Anthropic":
            model_name = st.selectbox(
                "Model",
                options=["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022", "claude-3-opus-20240229"],
                index=0
            )
        elif api_provider == "Google (Gemini)":
            model_name = st.selectbox(
                "Model",
                options=["gemini-2.0-flash-exp", "gemini-1.5-pro", "gemini-1.5-flash"],
                index=0
            )
        else:
            model_name = st.text_input("Model Name", value="gpt-4o-mini")
        
        temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1, help="Higher = more creative")
    
    # Substrate Configuration
    st.subheader("2. Verified Substrate (Ground Truth)")
    substrate_input_llm = st.text_area(
        "Enter verified facts (one per line)",
        value="The Eiffel Tower is in Paris\nWater boils at 100°C at sea level\nPython is a programming language",
        height=150,
        key="llm_test_substrate"
    )
    
    # Prompt Configuration
    st.subheader("3. Prompt Configuration")
    col_prompt1, col_prompt2 = st.columns([2, 1])
    
    with col_prompt1:
        user_prompt = st.text_area(
            "User Prompt",
            value="Tell me three facts about the topics mentioned in the context.",
            height=100,
            help="The prompt sent to the LLM"
        )
    
    with col_prompt2:
        num_responses = st.number_input(
            "Number of Responses",
            min_value=1,
            max_value=10,
            value=3,
            help="Generate multiple responses for scoring"
        )
        
        use_system_prompt = st.checkbox("Use System Prompt", value=True)
    
    if use_system_prompt:
        system_prompt = st.text_area(
            "System Prompt",
            value="You are a helpful assistant. Provide accurate, factual information.",
            height=80
        )
    else:
        system_prompt = None
    
    # Run Button
    if st.button("Test LLM with Guardrail", type="primary"):
        if not api_key:
            st.error("Please provide an API key")
        else:
            # Parse substrate
            substrate_list = [line.strip() for line in substrate_input_llm.split('\n') if line.strip()]
            
            if not substrate_list:
                st.error("Please provide substrate facts")
            else:
                with st.spinner(f"Generating {num_responses} response(s) from {api_provider}..."):
                    try:
                        # Import LLM client
                        responses = []
                        
                        if api_provider == "OpenAI":
                            import openai
                            client = openai.OpenAI(api_key=api_key)
                            
                            for i in range(num_responses):
                                messages = []
                                if system_prompt:
                                    messages.append({"role": "system", "content": system_prompt})
                                messages.append({"role": "user", "content": user_prompt})
                                
                                response = client.chat.completions.create(
                                    model=model_name,
                                    messages=messages,
                                    temperature=temperature
                                )
                                responses.append(response.choices[0].message.content)
                        
                        elif api_provider == "Anthropic":
                            import anthropic
                            client = anthropic.Anthropic(api_key=api_key)
                            
                            for i in range(num_responses):
                                response = client.messages.create(
                                    model=model_name,
                                    max_tokens=1024,
                                    temperature=temperature,
                                    system=system_prompt if system_prompt else "",
                                    messages=[{"role": "user", "content": user_prompt}]
                                )
                                responses.append(response.content[0].text)
                        
                        elif api_provider == "Google (Gemini)":
                            import google.generativeai as genai
                            genai.configure(api_key=api_key)
                            model = genai.GenerativeModel(model_name)
                            
                            for i in range(num_responses):
                                full_prompt = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt
                                response = model.generate_content(
                                    full_prompt,
                                    generation_config=genai.types.GenerationConfig(
                                        temperature=temperature,
                                    )
                                )
                                responses.append(response.text)
                        
                        else:  # Custom OpenAI-compatible
                            import openai
                            base_url = st.text_input("Base URL", value="https://api.openai.com/v1")
                            client = openai.OpenAI(api_key=api_key, base_url=base_url)
                            
                            for i in range(num_responses):
                                messages = []
                                if system_prompt:
                                    messages.append({"role": "system", "content": system_prompt})
                                messages.append({"role": "user", "content": user_prompt})
                                
                                response = client.chat.completions.create(
                                    model=model_name,
                                    messages=messages,
                                    temperature=temperature
                                )
                                responses.append(response.choices[0].message.content)
                        
                        st.success(f"✓ Generated {len(responses)} response(s)")
                        
                        # Now run the guardrail
                        st.markdown("---")
                        st.subheader("Guardrail Analysis")
                        
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
                            if result_text:
                                st.success(f"🟢 **Selected Response**")
                                st.markdown(f"> {result_text}")
                            else:
                                st.warning("🔴 **Abstained** - No responses met yield strength requirements")
                            
                            # Show all responses with scores
                            st.markdown("### All Responses with Scores")
                            
                            for i, (response, cand_vec) in enumerate(zip(responses, resp_vecs)):
                                cand_metrics = metrics['candidates'][i]
                                is_selected = (result_text == response)
                                
                                # Compute alignment to best substrate
                                best_alignment = -1
                                best_substrate = None
                                
                                for j, sub_vec in enumerate(sub_vecs):
                                    cos_sim = cosine_similarity(cand_vec, sub_vec)
                                    alignment = (cos_sim + 1.0) / 2.0
                                    if alignment > best_alignment:
                                        best_alignment = alignment
                                        best_substrate = substrate_list[j]
                                
                                # Display card
                                status = "🟢 SELECTED" if is_selected else ("🔴 EXCLUDED" if cand_metrics.get('fractured', False) else "⚪ SURVIVED")
                                
                                with st.expander(f"Response {i+1} - {status}", expanded=is_selected):
                                    st.markdown(f"**Response:**\n> {response}")
                                    st.markdown("---")
                                    
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Alignment Score", f"{best_alignment:.4f}")
                                    with col2:
                                        st.metric("Final Stress", f"{cand_metrics['stress']:.4f}")
                                    with col3:
                                        fractured = cand_metrics.get('fractured', False)
                                        st.metric("Status", "Fractured" if fractured else "Intact")
                                    
                                    st.caption(f"Best substrate match: *{best_substrate}*")
                                    
                                    # Show stress evolution
                                    phase_log = cand_metrics['phase_log']
                                    stress_data = [entry['stress'] for entry in phase_log]
                                    
                                    fig_stress = go.Figure()
                                    fig_stress.add_trace(go.Scatter(
                                        y=stress_data,
                                        mode='lines+markers',
                                        name='Stress',
                                        line=dict(color='red' if fractured else 'green')
                                    ))
                                    fig_stress.update_layout(
                                        title=f"Stress Evolution - Response {i+1}",
                                        yaxis_title="Cumulative Stress σ",
                                        xaxis_title="Step",
                                        height=250,
                                        template="plotly_dark"
                                    )
                                    st.plotly_chart(fig_stress, use_container_width=True)
                            
                            # Summary metrics
                            st.markdown("### Summary Metrics")
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
                            with st.expander("📊 Complete Numerical Audit Trail", expanded=False):
                                st.caption("Full vectors, distances, and stress evolution for all responses")
                                
                                numbers_view = {
                                    "llm_info": {
                                        "provider": api_provider,
                                        "model": model_name,
                                        "temperature": temperature,
                                        "num_responses": num_responses
                                    },
                                    "substrate": [
                                        {"text": t, "vec2": [round(v[0], 8), round(v[1], 8)]}
                                        for t, v in zip(substrate_list, sub_vecs)
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
                                        "comparisons": comparisons,
                                        "engine": {
                                            "fractured": cand_metrics.get('fractured', False),
                                            "fractured_step": cand_metrics.get('fractured_step'),
                                            "final_stress": round(float(cand_metrics['stress']), 8),
                                            "hash": cand_metrics.get('hash', 'N/A')
                                        }
                                    })
                                
                                st.json(numbers_view)
                    
                    except ImportError as e:
                        st.error(f"Missing library: {str(e)}")
                        st.info("Install required libraries:\n```\npip install openai anthropic google-generativeai\n```")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        st.exception(e)

# ============================================================================
# Footer
# ============================================================================

st.markdown("---")
st.caption("Development GUI | Deterministic Governance Mechanism")
