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


class EmbedderProtocol(Protocol):
    """Protocol for embedding providers"""
    def embed(self, text: str) -> Tuple[float, float]:
        """Convert text to 2D vector (x, y)"""
        ...


class MockEmbedder:
    """
    Deterministic string-to-2D-vector projection for demonstration.
    Uses SHA-256 to project text onto the unit square [0, 1] x [0, 1].
    
    NOTE: In production, this would be replaced by a dimensionality reduction
    pipeline (e.g., UMAP/PCA over BERT embeddings), but for the reference
    mechanism, we just need STABLE, DETERMINISTIC mapping.
    """
    
    def embed(self, text: str) -> Tuple[float, float]:
        """Project text deterministically to 2D space"""
        # Canonicalize text
        text = text.strip().lower()
        
        # Hash to get stable bits
        h = hashlib.sha256(text.encode('utf-8')).hexdigest()
        
        # Take first 8 chars for X, next 8 for Y
        # 8 hex chars = 32 bits = 4.29 billion values
        max_val = 16**8
        
        x_int = int(h[:8], 16)
        y_int = int(h[8:16], 16)
        
        # Normalize to [0, 1]
        x = x_int / max_val
        y = y_int / max_val
        
        return (x, y)


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
                 config_preset: str = 'balanced'):
        """
        Initialize the guardrail.
        
        Args:
            substrate_texts: List of verified factual strings
            embedder: Optional custom embedder (defaults to MockEmbedder)
            config_preset: Engine configuration preset
        """
        self.embedder = embedder or MockEmbedder()
        self.config = load_config(config_preset)
        
        # Initialize substrate
        self.substrate = VerifiedSubstrate(
            elastic_modulus_mode=self.config.get('elastic_modulus_mode', 'multiplicative'),
            elastic_modulus_sigma=self.config.get('elastic_modulus_sigma', 0.5)
        )
        
        # Embed and add substrate states
        for text in substrate_texts:
            x, y = self.embedder.embed(text)
            self.substrate.add_verified_state(Vector2D(x=x, y=y, properties=None))
            
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
            
        # Map candidates to vectors
        engine = MaterialFieldEngine(
            self.substrate,
            lambda_min=self.config['lambda_min'],
            lambda_max=self.config['lambda_max'],
            inference_steps=self.config['total_steps']
        )
        
        # Embed candidates
        vector_coords = []
        for c in candidates:
            vector_coords.append(self.embedder.embed(c))
            
        engine.initialize_candidates(vector_coords)
        
        # Run inference
        results = engine.run_inference()
        
        # Map result back to text
        final_vector = results.get('final_output')
        
        if final_vector and final_vector.candidate_index is not None:
            # Check if execution flagged it as grounded
            # (Double check hallucination_free flag from engine)
            if results.get('hallucination_free', False):
                return candidates[final_vector.candidate_index]
        
        return None
    
    def inspect(self, candidates: List[str]) -> Dict:
        """
        Run inference and return full inspection details.
        """
        engine = MaterialFieldEngine(
            self.substrate,
            lambda_min=self.config['lambda_min'],
            lambda_max=self.config['lambda_max'],
            inference_steps=self.config['total_steps']
        )
        
        vector_coords = []
        for c in candidates:
            vector_coords.append(self.embedder.embed(c))
            
        engine.initialize_candidates(vector_coords)
        results = engine.run_inference()
        
        return {
            'selected_text': candidates[results['final_output'].candidate_index] if results['final_output'] else None,
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
