#!/usr/bin/env python3
"""
Check which elements are supported by a MACE model.
Usage: python check_model_elements.py <model_path>
"""

import sys
import os
import torch
from ase.data import chemical_symbols

# Set environment variable for PyTorch 2.6+ compatibility
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'

def check_model_elements(model_path):
    """Check which elements are supported by a MACE model."""
    print(f"Loading model from: {model_path}")
    
    try:
        # Load the model with weights_only=False for PyTorch 2.6+ compatibility
        model = torch.load(model_path, map_location="cpu", weights_only=False)
        
        # Get atomic numbers from model
        if hasattr(model, 'atomic_numbers'):
            atomic_numbers = model.atomic_numbers
        elif hasattr(model, 'config') and hasattr(model.config, 'atomic_numbers'):
            atomic_numbers = model.config.atomic_numbers
        else:
            print("Warning: Could not find atomic_numbers in model")
            print("Available attributes:", dir(model))
            return
        
        # Convert atomic numbers to element symbols
        elements = [chemical_symbols[int(z)] for z in atomic_numbers]
        
        print(f"\nModel supports {len(elements)} elements:")
        print(f"Atomic numbers: {[int(z) for z in atomic_numbers]}")
        print(f"Elements: {elements}")
        
        # Check for Si (atomic number 14)
        si_supported = 14 in [int(z) for z in atomic_numbers]
        
        print(f"\n{'='*50}")
        if si_supported:
            print("✓ Si (Silicon) is SUPPORTED by this model")
        else:
            print("✗ Si (Silicon) is NOT supported by this model")
        print(f"{'='*50}")
        
        # Check for other elements in silicone-water system
        required_elements = {'Si': 14, 'O': 8, 'C': 6, 'H': 1}
        print("\nRequired elements for silicone-water system:")
        for elem, z in required_elements.items():
            supported = z in [int(z) for z in atomic_numbers]
            status = "✓" if supported else "✗"
            print(f"  {status} {elem} (Z={z})")
        
        all_supported = all(z in [int(z) for z in atomic_numbers] 
                           for z in required_elements.values())
        
        print(f"\n{'='*50}")
        if all_supported:
            print("✓ This model is COMPATIBLE with silicone-water system")
        else:
            print("✗ This model is NOT compatible with silicone-water system")
        print(f"{'='*50}")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python check_model_elements.py <model_path>")
        print("\nExamples:")
        print("  python check_model_elements.py mace_pretained_models/mace-mpa-0-medium.model")
        print("  python check_model_elements.py mace_pretained_models/MACE-OFF23_medium.model")
        sys.exit(1)
    
    model_path = sys.argv[1]
    check_model_elements(model_path)