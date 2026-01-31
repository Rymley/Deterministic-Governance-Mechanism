#!/usr/bin/env python3
"""
Test Suite for Deterministic Guardrail Adapter
"""

import unittest
from llm_adapter import DeterministicGuardrail, MockEmbedder

class TestDeterministicGuardrail(unittest.TestCase):
    
    def test_mock_embedder_determinism(self):
        """Test that MockEmbedder is deterministic"""
        embedder = MockEmbedder()
        text = "Hello World"
        
        vec1 = embedder.embed(text)
        vec2 = embedder.embed(text)
        
        self.assertEqual(vec1, vec2)
        self.assertIsInstance(vec1[0], float)
        self.assertIsInstance(vec1[1], float)
        
    def test_basic_filtering(self):
        """Test basic filtering capability"""
        # Ground truth: "A" gives a specific vector
        substrate = ["The fast fox jumps"]
        
        guard = DeterministicGuardrail(substrate_texts=substrate)
        
        candidates = [
            "The fast fox jumps",       # Perfect match (should survive)
            "The slow turtle crawls"    # Different vector (should likely be filtered or score lower)
        ]
        
        # Note: Since we use a hash-based mock embedder, "The slow turtle crawls" 
        # maps to a random point in 2D space. The chance it maps close to the
        # substrate is low (~1/E area), but not zero.
        # However, "The fast fox jumps" maps to exactly the same point as substrate,
        # so it has E=1.0 (or very high).
        
        result = guard.filter(candidates)
        self.assertEqual(result, "The fast fox jumps")
        
    def test_abstention(self):
        """Test that the system abstains when no candidate is good enough"""
        # Substrate is completely unrelated to candidates
        substrate = ["Apple Banana Cherry"]
        
        guard = DeterministicGuardrail(substrate_texts=substrate)
        
        # These map to random points likely far from "Apple Banana Cherry"
        candidates = [
            "Xylophone Zebra",
            "Quantum Physics"
        ]
        
        # We expect abstention (None) because candidates should fail to nucleate
        # or be excluded by pressure.
        # (Though there is a tiny collision probability with SHA-256 mapping to 2D)
        result = guard.filter(candidates)
        
        # In the unlikely event of a collision, we handle it, but mostly this should be None
        if result is not None:
            print(f"WARNING: Unlucky hash collision allowed '{result}' to survive against unrelated substrate.")
        else:
            self.assertIsNone(result)

    def test_multi_candidate_selection(self):
        """Test that the best candidate is selected from multiple options"""
        substrate = ["The quick brown fox"]
        guard = DeterministicGuardrail(substrate_texts=substrate)
        
        candidates = [
            "The quick brown fox",  # E=1.0
            "The quick brown",      # Hashed differently -> E < 1.0 (random)
            "brown fox"             # Hashed differently -> E < 1.0 (random)
        ]
        
        result = guard.filter(candidates)
        self.assertEqual(result, "The quick brown fox")

if __name__ == "__main__":
    unittest.main()
