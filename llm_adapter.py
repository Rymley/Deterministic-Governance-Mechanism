#!/usr/bin/env python3
"""
LLM Adapter for Deterministic Governance Mechanism
Bridges high-dimensional text/LLM outputs to the 2D material field engine.

A model-agnostic post-processor that evaluates candidate outputs against a declared substrate.
It deterministically accepts, rejects, or abstains based on explicit constraints.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Any, Dict, Protocol
import hashlib
import json

from material_field_engine import (
    MaterialFieldEngine,
    VerifiedSubstrate,
    Vector2D,
    load_config
)


_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "else", "when", "while",
    "of", "to", "in", "on", "at", "by", "for", "with", "about", "as", "into",
    "is", "are", "was", "were", "be", "being", "been", "do", "does", "did",
    "this", "that", "these", "those", "it", "its", "it's", "you", "your",
    "i", "we", "they", "them", "he", "she", "his", "her", "their", "ours",
    "from", "up", "down", "over", "under", "again", "further", "here", "there",
    "why", "how", "what", "which", "who", "whom"
}


def _tokenize(text: str) -> List[str]:
    cleaned = []
    for ch in text.lower():
        cleaned.append(ch if ch.isalnum() else " ")
    tokens = [tok for tok in "".join(cleaned).split() if tok and tok not in _STOPWORDS]
    return tokens


def _token_set(texts: List[str]) -> set:
    tokens = set()
    for t in texts:
        tokens.update(_tokenize(t))
    return tokens


class DeterministicHashEmbedderND:
    """
    Deterministic hash-based embedder producing N-D vectors in [0,1].
    """

    def __init__(self, dims: int = 16):
        if dims < 2:
            raise ValueError("dims must be >= 2")
        self.dims = dims

    def embed(self, text: str) -> List[float]:
        text = text.strip().lower()
        values: List[float] = []
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        cursor = 0

        for i in range(self.dims):
            if cursor + 2 > len(digest):
                digest = hashlib.sha256(digest + i.to_bytes(2, "big")).digest()
                cursor = 0
            chunk = digest[cursor:cursor + 2]
            cursor += 2
            val = int.from_bytes(chunk, "big") / 65535.0
            values.append(val)

        return values

    def project_2d(self, vector: List[float]) -> Tuple[float, float]:
        if len(vector) < 2:
            return (0.0, 0.0)
        return (float(vector[0]), float(vector[1]))


class EmbedderProtocol(Protocol):
    """Protocol for embedding providers"""
    def embed(self, text: str) -> List[float]:
        """Convert text to a vector (length >= 2)"""
        ...


class MockEmbedder(DeterministicHashEmbedderND):
    """
    Deterministic 2D hash embedder for legacy demos.
    """

    def __init__(self):
        super().__init__(dims=2)


@dataclass
class ContentObject:
    """Container for content and its physical representation"""
    text: str
    vector: Vector2D
    metadata: Dict[str, Any] = None


class DeterministicGuardrail:
    """
    Guardrail wrapper for LLM outputs.
    Applies mechanical exclusion to filter hallucinations.
    """
    
    def __init__(self, 
                 substrate_texts: List[str],
                 embedder: Optional[EmbedderProtocol] = None,
                 config_preset: str = 'balanced',
                 topic_gate_min_overlap: int = 1,
                 topic_gate_enabled: bool = True,
                 ambiguity_detection_enabled: bool = True):
        """
        Initialize the guardrail.
        
        Args:
            substrate_texts: List of verified factual strings
            embedder: Optional custom embedder (defaults to MockEmbedder)
            config_preset: Engine configuration preset
        """
        self.embedder = embedder or DeterministicHashEmbedderND()
        self.config = load_config(config_preset)
        self.topic_gate_min_overlap = topic_gate_min_overlap
        self.topic_gate_enabled = topic_gate_enabled
        self.ambiguity_detection_enabled = ambiguity_detection_enabled
        self.substrate_tokens = _token_set(substrate_texts)
        
        # Initialize substrate
        self.substrate = VerifiedSubstrate(
            elastic_modulus_mode=self.config.get('elastic_modulus_mode', 'multiplicative'),
            elastic_modulus_sigma=self.config.get('elastic_modulus_sigma', 0.5)
        )
        
        # Embed and add substrate states
        for text in substrate_texts:
            vec = self.embedder.embed(text)
            self.substrate.add_verified_state(
                Vector2D(x=vec[0], y=vec[1], properties=None, coords=vec)
            )

    def _passes_topic_gate(self, text: str) -> bool:
        if not self.topic_gate_enabled:
            return True
        tokens = set(_tokenize(text))
        if not tokens:
            return False
        return len(tokens & self.substrate_tokens) >= self.topic_gate_min_overlap

    def _is_clarification(self, text: str) -> bool:
        """Detect if the response is asking for clarification due to ambiguity"""
        if not self.ambiguity_detection_enabled:
            return False
            
        clarification_markers = [
            "could you please specify",
            "please specify",
            "please clarify",
            "what do you mean",
            "can you provide",
            "no specific topics",
            "no topics were mentioned",
            "which topic"
        ]
        text_lower = text.lower()
        return any(marker in text_lower for marker in clarification_markers)
            
    def filter(self, candidates: List[str]) -> Optional[str]:
        """
        Filter a list of candidate outputs (e.g., from an LLM beam search).
        Returns the single surviving "factual" string, or None if all are excluded.
        
        Args:
            candidates: List of candidate strings from the LLM
            
        Returns:
            The best surviving candidate string, or None (abstention)
        """
        if not candidates:
            return None

        topic_ok = [self._passes_topic_gate(c) for c in candidates]
        allowed = [(idx, c) for idx, c in enumerate(candidates) if topic_ok[idx]]
        if not allowed:
            return None

        engine = MaterialFieldEngine(
            self.substrate,
            lambda_min=self.config['lambda_min'],
            lambda_max=self.config['lambda_max'],
            inference_steps=self.config['total_steps']
        )

        vector_coords = []
        allowed_indices = []
        for idx, c in allowed:
            vec = self.embedder.embed(c)
            vector_coords.append(vec)
            allowed_indices.append(idx)

        engine.initialize_candidates(vector_coords)
        results = engine.run_inference()

        final_vector = results.get('final_output')
        if final_vector and final_vector.candidate_index is not None:
            if results.get('hallucination_free', False):
                original_idx = allowed_indices[final_vector.candidate_index]
                return candidates[original_idx]

        return None
    
    def inspect(self, candidates: List[str]) -> Dict:
        """
        Run inference and return full inspection details.
        """
        topic_ok = [self._passes_topic_gate(c) for c in candidates]
        is_clarification = [self._is_clarification(c) for c in candidates]
        
        # Allowed for physics: Passes topic gate AND is not a clarification (clarification bypasses physics)
        allowed_indices = [
            i for i, ok in enumerate(topic_ok) 
            if ok and not is_clarification[i]
        ]
        allowed_texts = [candidates[i] for i in allowed_indices]

        if not allowed_texts:
            candidate_metrics = [
                {
                    'phase_log': [],
                    'fractured': True,
                    'fractured_step': 0,
                    'stress': 0.0,
                    'hash': None,
                    'out_of_domain': True,
                }
                for _ in candidates
            ]
            return {
                'selected_text': None,
                'metrics': {
                    'final_output': None,
                    'phase_log': [],
                    'total_excluded': len(candidates),
                    'latency_ms': 0.0,
                    'latency_per_step_ms': 0.0,
                    'latency_ns': 0,
                    'latency_per_step_ns': 0,
                    'deterministic': True,
                    'hallucination_free': True,
                    'abstained': True,
                    'final_stress_q': None,
                    'final_stress': None,
                    'max_stress_q': 0,
                    'max_stress': 0.0,
                    'candidates': candidate_metrics,
                    'topic_gate_excluded': len(candidates),
                }
            }

        engine = MaterialFieldEngine(
            self.substrate,
            lambda_min=self.config['lambda_min'],
            lambda_max=self.config['lambda_max'],
            inference_steps=self.config['total_steps']
        )

        vector_coords = []
        for c in allowed_texts:
            vector_coords.append(self.embedder.embed(c))

        engine.initialize_candidates(vector_coords)
        results = engine.run_inference(collect_trace=True)

        final_vector = results.get('final_output')
        selected_text = None
        if final_vector and final_vector.candidate_index is not None:
            original_idx = allowed_indices[final_vector.candidate_index]
            final_vector.candidate_index = original_idx
            selected_text = candidates[original_idx]

        allowed_metrics = results.get('candidates', [])
        allowed_map = {orig_idx: j for j, orig_idx in enumerate(allowed_indices)}
        candidate_metrics = []
        for i in range(len(candidates)):
            if not topic_ok[i]:
                candidate_metrics.append({
                    'phase_log': [],
                    'fractured': True,
                    'fractured_step': 0,
                    'stress': 0.0,
                    'hash': None,
                    'out_of_domain': True,
                })
                continue
            
            # 2. Check Clarification (Bypass Physics, Auto-Pass)
            if is_clarification[i]:
                candidate_metrics.append({
                    'phase_log': [],
                    'fractured': False,
                    'fractured_step': None,
                    'stress': 0.0,
                    'hash': None,
                    'out_of_domain': False,
                    'is_clarification': True
                })
                continue

            mapped_idx = allowed_map[i]
            entry = allowed_metrics[mapped_idx]
            entry['out_of_domain'] = False
            candidate_metrics.append(entry)

        results['candidates'] = candidate_metrics
        results['total_excluded'] = results.get('total_excluded', 0) + (len(candidates) - len(allowed_indices))
        results['topic_gate_excluded'] = len(candidates) - len(allowed_indices)

        return {
            'selected_text': selected_text,
            'metrics': results
        }


def demo_simple():
    """Simple demonstration of the adapter"""
    print("Initializing Deterministic Guardrail...")
    
    # 1. Define Ground Truth (Substrate)
    # The system knows these facts are true.
    facts = [
        "The sky is blue",
        "Water is wet",
        "Paris is the capital of France"
    ]
    
    guard = DeterministicGuardrail(substrate_texts=facts)
    
    print(f"Substrate loaded with {len(facts)} verified facts.")
    
    # 2. Test Cases
    scenarios = [
        {
            "name": "Factual Consistency",
            "candidates": ["The sky is blue", "The sky is green"]
        },
        {
            "name": "Hallucination Check",
            "candidates": ["The moon is made of cheese", "The moon is made of rock"]
        },
        {
            "name": "Subtle Error",
            "candidates": ["Paris is in Germany", "Paris is in France"]
        }
    ]
    
    print("\nRunning Scenarios:\n")
    
    for sc in scenarios:
        print(f"--- Scenario: {sc['name']} ---")
        print(f"Candidates: {sc['candidates']}")
        
        result = guard.filter(sc['candidates'])
        
        if result:
            print(f"✅ PASSED: Selected '{result}'")
        else:
            print(f"🛡️ ABSTAINED: All candidates excluded (Prevention mode)")
        print()


if __name__ == "__main__":
    demo_simple()
