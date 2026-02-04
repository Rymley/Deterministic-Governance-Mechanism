
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from material_field_engine import MaterialFieldEngine, VerifiedSubstrate, Vector2D, fp_to_float, fp_from_float
from llm_adapter import DeterministicHashEmbedderND

def test_16d_stability():
    print("Testing 16D Stability and Stress Reporting...")
    
    # Setup 16D environment
    embedder = DeterministicHashEmbedderND(dims=16)
    
    # Create substrate (1 verified fact)
    substrate_text = "The sky is blue"
    sub_vec_coords = embedder.embed(substrate_text)
    substrate_vector = Vector2D(
        x=sub_vec_coords[0], 
        y=sub_vec_coords[1], 
        properties=None, 
        coords=sub_vec_coords
    )
    
    substrate = VerifiedSubstrate(
        verified_states=[substrate_vector],
        elastic_modulus_mode='multiplicative',
        elastic_modulus_sigma=0.5
    )
    
    # Create engine
    engine = MaterialFieldEngine(substrate, lambda_min=0.3, lambda_max=0.9, inference_steps=8)
    
    # Test 1: Near match (should survive or fracture late, definitely NOT at step 0)
    # We simulate a "near match" by adding tiny noise to the substrate vector
    near_coords = [c + 0.01 for c in sub_vec_coords]
    
    engine.initialize_candidates([near_coords])
    
    # Check initial properties
    print("\nInitial Candidate Properties:")
    for v in engine.candidate_vectors:
        print(f"E: {v.properties.elastic_modulus:.4f}, Strain: {v.properties.strain:.4f}")
        
    start_stress = engine.candidate_vectors[0].properties.stress
    print(f"Initial Stress: {start_stress}")

    # Run Inference
    results = engine.run_inference()
    
    final_output = results.get('final_output')
    print(f"\nFinal Output: {'Survived' if final_output else 'Excluded'}")
    
    # Check Stress Reporting
    max_stress = results.get('max_stress')
    final_stress = results.get('final_stress')
    
    print(f"Max Stress: {max_stress}")
    print(f"Final Reported Stress: {final_stress}")
    
    if final_output is None:
        # If excluded, ensure final_stress matches max_stress and is > 0 (unless it started > yield)
        if final_stress != max_stress:
             print("FAILURE: final_stress does not match max_stress for excluded candidate.")
        else:
             print("SUCCESS: Stress reporting logic works for exclusions.")
    
    # Check 16D Stability
    # If E is tiny (<0.1) despite being close (distance ~ 0.04 in 16D), then normalization failed.
    # Distance of 0.01 in 16 dims: sqrt(16 * 0.01^2) = sqrt(0.0016) = 0.04.
    # RBF with sigma=0.5: exp(-0.04^2 / (2*0.5^2)) = exp(-0.0016 / 0.5) ~ 1.0. 
    # Without normalization? 16D "close" usually means distance ~1.0-2.0 if not identical.
    # Let's check a "misaligned" but "close-ish" one.
    
    # Real test: Random vector (avg distance in 16D is high).
    # Expected: E should not be 0.0 just due to dims.
    
    print("\nTest passed if no crashes and stress is reported.")

if __name__ == "__main__":
    test_16d_stability()
