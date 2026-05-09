#!/usr/bin/env python3
"""
Convert MACE model to MLIAP format for LAMMPS.
Based on the reference script in test_mace2_multiple-walkers_NewImage

Supports optional cuEquivariance acceleration if the library is installed.

Usage:
    python convert_mace_to_mliap.py [model_path] [output_path]

Examples:
    python convert_mace_to_mliap.py
        # Uses default: mace_pretained_models/mace-mpa-0-medium.model
    
    python convert_mace_to_mliap.py /path/to/pretrained_model.model
        # Uses specified model
    
    python convert_mace_to_mliap.py model.model output.model
        # Uses specified model and output path
"""

import os
import sys
from pathlib import Path
import torch

# Set environment variables
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'

# Check for cuEquivariance availability
try:
    from mace.cli.convert_e3nn_cueq import run as run_e3nn_to_cueq
    CUEQ_AVAILABLE = True
    print("cuEquivariance conversion is available - GPU acceleration will be enabled!")
except (ImportError, ModuleNotFoundError) as e:
    CUEQ_AVAILABLE = False
    print(f"cuEquivariance conversion not available: {e}")
    print("Model will use standard e3nn operations (still runs on GPU via PyTorch)")

def convert_model_to_mliap(model_path, output_path=None):
    """
    Convert MACE model to MLIAP format for LAMMPS.
    
    Args:
        model_path: Path to the MACE model file (.model)
        output_path: Output path (default: model_path + '-mliap_lammps.pt')
    """
    if output_path is None:
        output_path = model_path + '-mliap_lammps.pt'
    
    print(f"Loading model from: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.load(
        model_path,
        map_location=torch.device(device),
    )
    
    # Try to convert to cuEquivariance for GPU acceleration
    # IMPORTANT: Do NOT pass device="cuda" to run_e3nn_to_cueq() because:
    # - When device="cuda", conv_fusion=True is set
    # - conv_fusion optimization is incompatible with LAMMPS MLIAP interface
    # - MACE official create_lammps_model.py also uses default device="cpu"
    # The model will still use cuEquivariance optimized operations (without conv_fusion)
    if CUEQ_AVAILABLE:
        print("Converting model to cuEquivariance format for GPU acceleration...")
        try:
            import copy
            # Use copy.deepcopy like official MACE script, and NO device parameter
            model = run_e3nn_to_cueq(copy.deepcopy(model))
            print("cuEquivariance conversion successful!")
        except Exception as e:
            print(f"cuEquivariance conversion failed: {e}")
            print("Falling back to standard e3nn model")
    
    # Convert to float64 and move to CPU for LAMMPS compatibility
    model = model.double().to("cpu")
    print("Model loaded and converted to float64")
    
    # Set MLIAP flag
    model.lammps_mliap = True
    
    # Import MACE LAMMPS interface
    try:
        from mace.calculators.lammps_mliap_mace import LAMMPS_MLIAP_MACE
    except ImportError as e:
        print("Error: Could not import LAMMPS_MLIAP_MACE from mace.calculators.lammps_mliap_mace")
        print("Please ensure MACE is properly installed and in your Python path.")
        raise
    
    # Get head (use last head if multiple heads exist)
    if hasattr(model, 'heads'):
        heads = model.heads
        if len(heads) == 1:
            head = heads[0]
            print(f"Using head: {head}")
        else:
            head = heads[-1]  # Use last head
            print(f"Multiple heads found. Using last head: {head}")
    else:
        head = None
        print("No heads found in model, proceeding without head specification")
    
    # Create MLIAP wrapper
    if head is not None:
        lammps_model = LAMMPS_MLIAP_MACE(model, head=head)
    else:
        lammps_model = LAMMPS_MLIAP_MACE(model)
    
    # Save model
    torch.save(lammps_model, output_path)
    print(f"Model saved to: {output_path}")
    print("Conversion successful!")
    
    return output_path

if __name__ == '__main__':
    # Default model path: mace_pretained_models/mace-mpa-0-medium.model
    script_dir = Path(__file__).parent
    default_model_path = script_dir / "mace_pretained_models" / "mace-mpa-0-medium.model"
    
    if len(sys.argv) < 2:
        # Use default model if no argument provided
        if default_model_path.exists():
            print(f"No model path provided, using default: {default_model_path}")
            model_path = str(default_model_path)
            output_path = None
        else:
            print("Usage: convert_mace_to_mliap.py [model_path] [output_path]")
            print("Example: convert_mace_to_mliap.py /path/to/model.model")
            print(f"\nDefault model not found at: {default_model_path}")
            print("Please provide a model path or download the default model first.")
            sys.exit(1)
    else:
        model_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        convert_model_to_mliap(model_path, output_path)
    except Exception as e:
        print(f"Error during conversion: {e}", file=sys.stderr)
        sys.exit(1)

