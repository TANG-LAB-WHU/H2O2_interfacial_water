#!/usr/bin/env python3
"""
Script to build water-silicone interface structures for H2O2 mechanism studies.

This script supports multiple water placement modes, interfacial water layer control,
density-based packing via PACKMOL/mdapackmol, and vacuum gap for DFT slab calculations.

Features:
    - Multiple placement modes: random, grid, packmol, interface
    - Interfacial water layer with controlled orientation (dangling OH)
    - Density control via mdapackmol integration
    - Vacuum gap for DFT slab calculations
    - Multiple output formats: xyz, cp2k
    - H-bond pre-optimization option

Usage:
    # Basic usage with default parameters (interface mode)
    python build_water_on_silicone.py --n_water 50 --auto_interface --add_vacuum --water_layer_height 10.0  --vacuum_thickness 36.0

    # For AIMD/DFT studies with auto-calculated interface parameters
    python build_water_on_silicone.py --n_water 30 --auto_interface --add_vacuum

    # For large-scale MD with PACKMOL density control
    python build_water_on_silicone.py --n_water 200 --placement_mode packmol `
        --target_density 1.0

    # Custom output directory
    python build_water_on_silicone.py --n_water 50 --output_dir ./my_models

Default Parameters:
    --n_water               100          Number of water molecules
    --placement_mode        interface    Water placement mode
    --seed                  42           Random seed
    --z_offset              2.0 Å        Distance above slab surface
    --water_layer_height    15.0 Å       Water layer height
    --min_water_distance    2.5 Å        Minimum O-O distance
    --slab_z_min            1.0 Å        PDMS bottom z coordinate
    --interface_layer_thickness  5.0 Å   Interface layer thickness
    --interface_water_orientation  dangling  Interface water orientation
    --auto_interface        False        Auto-calculate interface params
    --surface_type          hydrophobic  Surface type for auto-calculation
    --n_interface_layers    2            Number of interface layers
    --target_density        1.0 g/cm³    Target water density (packmol)
    --add_vacuum            False        Add vacuum gap
    --vacuum_thickness      30.0 Å       Vacuum gap thickness
    --output_dir            output_model Output directory
    --output_formats        xyz,cp2k     Output formats
    --save_log              True         Save run log with timestamp

Author: Siqi Tang, siqit@outlook.com

"""

import argparse
import os
import sys
import subprocess
import tempfile
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Optional
import io

from typing import List, Tuple, Optional

try:
    from ase import Atoms
    from ase.io import read, write
    from ase.build import molecule
    from ase.geometry import get_distances
except ImportError:
    raise ImportError("ASE (Atomic Simulation Environment) is required. "
                      "Install it with: pip install ase")

# Optional: mdapackmol for density-based packing
try:
    import MDAnalysis as mda
    from mdapackmol import PackmolStructure, packmol
    HAS_MDAPACKMOL = True
except ImportError:
    HAS_MDAPACKMOL = False
    print("Note: mdapackmol not available. Install with: pip install mdapackmol")
    print("      PACKMOL placement mode will be disabled.")


# ============================================================================
# Constants for Interface Layer Calculation
# ============================================================================
# Water molecular properties
WATER_MOLAR_MASS = 18.015  # g/mol
AVOGADRO = 6.022e23
WATER_MOLECULE_DIAMETER = 2.8  # Å (van der Waals diameter)
WATER_MOLECULE_AREA = 3.1 ** 2  # Å² (effective molecular footprint ~9.6 Å²)
# ============================================================================
# Literature References
# ============================================================================
# [Ref 1] Patel et al., J. Phys. Chem. B, 2010, "Fluctuations of Water near Extended
#         Hydrophobic and Hydrophilic Surfaces"
#         https://pubs.acs.org/doi/10.1021/jp909048f
# [Ref 2] Davis et al., Nature, 2012, "Water structural transformation at molecular
#         hydrophobic interfaces"
#         https://www.nature.com/articles/nature11570
# [Ref 3] Godawat et al., PNAS, 2009, "Characterizing hydrophobicity of interfaces"
#         https://www.pnas.org/doi/10.1073/pnas.0902194106
# [Ref 4] Fellows et al., Langmuir, 2024, "How Thick is the Air–Water Interface?"
#         https://pubs.acs.org/doi/10.1021/acs.langmuir.4c02571
# [Ref 5] Shuler et al., J. Phys. Chem. B, 2004, "Surfactant-Mediated Decane−Water Interface"
#         https://pubs.acs.org/doi/10.1021/jp048773n
# [Ref 6] Sendner et al., Langmuir, 2009, "Interfacial Water at Hydrophobic Surfaces"
#         https://pubs.acs.org/doi/10.1021/la700561q
# [Ref 7] Chem. Sci., 2023, "Structural features of interfacial water"
#         https://pubs.rsc.org/en/content/articlelanding/2023/sc/d2sc02856e
# ============================================================================

# Water density parameters
# Used refs: [Ref 1] density fluctuations, [Ref 2] interface density, [Ref 3] depletion
WATER_DENSITY = {
    'bulk': 1.0,           # g/cm³, bulk water density
    'interface': {
        'hydrophobic': {
            'first_layer': 0.7,   # g/cm³ [Ref 1,2] reduced at hydrophobic interface
            'second_layer': 0.85, # g/cm³ [Ref 1] slightly reduced
            'depletion_zone': 0.3 # g/cm³ [Ref 3] near-surface depletion
        },
        'hydrophilic': {
            'first_layer': 1.1,   # g/cm³ [Ref 1] enhanced at hydrophilic interface
            'second_layer': 1.0,
        }
    }
}

# Interface layer parameters
# Used refs: [Ref 4] correlation_length, [Ref 5] 90-90 criterion, [Ref 6] depletion layer
INTERFACE_LAYER_PARAMS = {
    'hydrophobic': {
        'depletion_layer': 0.2,       # nm [Ref 6] density depletion layer thickness
        'monolayer': 2.8,             # Å, single water layer
        'bilayer': 5.6,               # Å, double water layer
        'structured_layer': 8.0,      # Å [Ref 7] structured water region
        'total_effect_range': 15.0,   # Å [Ref 1] total interfacial effect
        'correlation_length': 6.0,    # Å [Ref 4] structural anisotropy decay
        'density_ratio': 0.7,         # [Ref 1,2,3] interface/bulk density ratio
    },
    'hydrophilic': {
        'monolayer': 2.8,
        'bilayer': 5.6,
        'total_effect_range': 3.0,
        'density_ratio': 1.1,         # [Ref 1] interface/bulk density ratio (enhanced)
    }
}




# ============================================================================
# Interface Layer Calculation Functions
# ============================================================================
def calculate_interface_layer_params(
    surface_area: float,
    surface_type: str = 'hydrophobic',
    n_layers: int = 2,
    target_density: float = 1.0
) -> dict:
    """
    Calculate interface layer parameters based on surface area and water physics.
    
    This function uses literature values for hydrophobic/hydrophilic interfaces
    to automatically determine:
    - Interface layer thickness
    - Maximum number of interface water molecules
    - Recommended z_offset
    
    Args:
        surface_area: Surface area in Å²
        surface_type: 'hydrophobic' or 'hydrophilic'
        n_layers: Number of interface water layers (1=monolayer, 2=bilayer)
        target_density: Target water density in g/cm³
        
    Returns:
        Dictionary with calculated parameters:
        - interface_thickness: Å
        - n_interface_max: maximum interface waters
        - z_offset: recommended offset from surface
        - correlation_length: structural decay length
        
    References:
        - 1.5-2 nm interface on hydrophobic surfaces, from Patel et al., 
          J. Phys. Chem. B (2010): https://pubs.acs.org/doi/10.1021/jp909048f
        - 90-90 criterion for interface thickness, from Shuler et al., 
          J. Phys. Chem. B (2004): https://pubs.acs.org/doi/10.1021/jp048773n
        - 6-8 Å structural anisotropy decay, from Fellows et al., 
          Langmuir (2024): https://pubs.acs.org/doi/10.1021/acs.langmuir.4c02571
    """


    params = INTERFACE_LAYER_PARAMS.get(surface_type, INTERFACE_LAYER_PARAMS['hydrophobic'])
    
    # Calculate interface layer thickness based on number of layers
    if n_layers == 1:
        interface_thickness = params['monolayer']
    elif n_layers == 2:
        interface_thickness = params['bilayer']
    else:
        # For more layers, use water diameter spacing
        interface_thickness = n_layers * WATER_MOLECULE_DIAMETER
    
    # Calculate maximum number of interface water molecules
    # Based on surface packing: each water occupies ~WATER_MOLECULE_AREA
    n_interface_per_layer = int(surface_area / WATER_MOLECULE_AREA)
    n_interface_max = n_interface_per_layer * n_layers
    
    # Recommended z_offset (depletion layer for hydrophobic)
    if surface_type == 'hydrophobic':
        z_offset = params.get('depletion_layer', 0.2) * 10  # nm to Å
    else:
        z_offset = 1.5  # Å, minimal gap for hydrophilic
    
    # Correlation length
    correlation_length = params.get('correlation_length', 6.0)
    
    # Get density ratio for interface layer
    density_ratio = params.get('density_ratio', 1.0)
    
    # Get interface density from WATER_DENSITY dict
    interface_density_info = WATER_DENSITY['interface'].get(surface_type, {})
    bulk_density = WATER_DENSITY['bulk']
    
    # Adjust n_interface_max based on interface density (lower packing for hydrophobic)
    n_interface_max_adjusted = int(n_interface_max * density_ratio)
    
    result = {
        'interface_thickness': interface_thickness,
        'n_interface_max': n_interface_max_adjusted,
        'n_interface_per_layer': int(n_interface_per_layer * density_ratio),
        'z_offset': z_offset,
        'correlation_length': correlation_length,
        'n_layers': n_layers,
        'surface_type': surface_type,
        'density_ratio': density_ratio,
        'bulk_density': bulk_density,
        'interface_density': bulk_density * density_ratio,
    }
    
    return result


def print_interface_params(params: dict) -> None:
    """Print calculated interface layer parameters."""
    print("\n  Auto-calculated interface parameters:")
    print(f"    Surface type: {params['surface_type']}")
    print(f"    Number of layers: {params['n_layers']}")
    print(f"    Interface thickness: {params['interface_thickness']:.2f} Å")
    print(f"    Max interface waters: {params['n_interface_max']} "
          f"({params['n_interface_per_layer']} per layer)")
    print(f"    Recommended z_offset: {params['z_offset']:.2f} Å")
    print(f"    Correlation length: {params['correlation_length']:.2f} Å")
    print(f"    Density ratio (interface/bulk): {params['density_ratio']:.2f}")
    print(f"    Interface density: {params['interface_density']:.2f} g/cm³ "
          f"(bulk: {params['bulk_density']:.2f} g/cm³)")



# ============================================================================
# Header Replacement Functions
# ============================================================================
def extract_header_lines(filepath: str, n_lines: int = 2) -> list:
    """Extract the first n lines from a file."""
    with open(filepath, 'r') as f:
        lines = [f.readline() for _ in range(n_lines)]
    return lines


def replace_header_and_save(source_header_file: str, 
                             target_data_file: str, 
                             output_file: str) -> None:
    """
    Replace the header (first two lines) of target_data_file with header from 
    source_header_file and save to output_file.
    """
    header_lines = extract_header_lines(source_header_file, 2)
    
    with open(target_data_file, 'r') as f:
        all_lines = f.readlines()
    
    # Check column count and adjust header if needed
    data_line = all_lines[2].strip().split()
    n_cols = len(data_line)
    
    header_line = header_lines[1]
    if n_cols == 4 and 'tags:I:1' in header_line:
        header_line = header_line.replace(':tags:I:1', '')
        print("  Note: Removed 'tags' property from header (data has 4 columns)")
    
    all_lines[0] = header_lines[0]
    all_lines[1] = header_line
    
    if not all_lines[1].endswith('\n'):
        all_lines[1] += '\n'
    
    with open(output_file, 'w') as f:
        f.writelines(all_lines)
    
    print(f"  Relaxed structure saved to: {output_file}")


# ============================================================================
# Water Orientation Functions
# ============================================================================
def create_oriented_water(position: np.ndarray, 
                          orientation: str = 'random',
                          surface_normal: np.ndarray = np.array([0, 0, 1])) -> Atoms:
    """
    Create a water molecule with controlled orientation.
    
    Args:
        position: Center position (O atom location)
        orientation: 'random', 'dangling', 'parallel'
            - 'dangling': One OH points toward surface (negative z)
            - 'parallel': HOH plane parallel to surface
            - 'random': Random orientation
        surface_normal: Normal vector of the surface (default: [0,0,1])
    
    Returns:
        ASE Atoms object for oriented water molecule
    """
    water = molecule('H2O')
    
    if orientation == 'dangling':
        # Rotate so one OH bond points toward surface (opposite to surface normal)
        # Water molecule in ASE: O at center, two H atoms
        # We want one H to point downward (-z direction)
        water.rotate(180, 'y')  # Flip water upside down
        water.rotate(np.random.uniform(0, 360), 'z')  # Random azimuthal
        # Add small perturbation to tilt angle
        water.rotate(np.random.uniform(-20, 20), 'x')
        
    elif orientation == 'parallel':
        # Rotate so HOH plane is parallel to surface
        water.rotate(90, 'x')  # HOH plane now in xy
        water.rotate(np.random.uniform(0, 360), 'z')  # Random azimuthal
        
    else:  # random
        water.rotate(np.random.uniform(0, 360), 'x')
        water.rotate(np.random.uniform(0, 360), 'y')
        water.rotate(np.random.uniform(0, 360), 'z')
    
    water.translate(position)
    return water


# ============================================================================
# Water Placement Functions
# ============================================================================
def place_water_random(cell: np.ndarray, 
                       n_water: int,
                       z_start: float,
                       z_end: float,
                       min_distance: float = 2.5,
                       orientation: str = 'random',
                       seed: int = 42) -> List[Atoms]:
    """Random water placement with minimum distance constraint."""
    np.random.seed(seed)
    
    lx, ly = cell[0, 0], cell[1, 1]
    water_molecules = []
    positions = []
    
    max_attempts = n_water * 1000
    attempts = 0
    
    while len(positions) < n_water and attempts < max_attempts:
        x = np.random.uniform(0, lx)
        y = np.random.uniform(0, ly)
        z = np.random.uniform(z_start, z_end)
        new_pos = np.array([x, y, z])
        
        # Check minimum distance
        is_valid = True
        for existing_pos in positions:
            dx = abs(new_pos[0] - existing_pos[0])
            dy = abs(new_pos[1] - existing_pos[1])
            dz = abs(new_pos[2] - existing_pos[2])
            dx = min(dx, lx - dx)
            dy = min(dy, ly - dy)
            dist = np.sqrt(dx**2 + dy**2 + dz**2)
            if dist < min_distance:
                is_valid = False
                break
        
        if is_valid:
            positions.append(new_pos)
            water = create_oriented_water(new_pos, orientation)
            water_molecules.append(water)
        
        attempts += 1
    
    if len(positions) < n_water:
        print(f"  Warning: Only placed {len(positions)}/{n_water} water molecules")
    
    return water_molecules


def place_water_grid(cell: np.ndarray,
                     n_water: int,
                     z_start: float,
                     z_end: float,
                     orientation: str = 'random',
                     perturbation: float = 0.3,
                     seed: int = 42) -> List[Atoms]:
    """Grid-based water placement with random perturbation."""
    np.random.seed(seed)
    
    lx, ly = cell[0, 0], cell[1, 1]
    lz = z_end - z_start
    
    # Calculate grid spacing based on water density
    # For 1 g/cm³, water spacing is about 3.1 Å
    water_spacing = 3.1  # Å
    
    nx = max(1, int(lx / water_spacing))
    ny = max(1, int(ly / water_spacing))
    nz = max(1, int(lz / water_spacing))
    
    # Adjust to fit requested number
    n_grid = nx * ny * nz
    if n_grid < n_water:
        # Increase grid density
        factor = (n_water / n_grid) ** (1/3)
        nx = int(nx * factor) + 1
        ny = int(ny * factor) + 1
        nz = int(nz * factor) + 1
    
    dx = lx / nx
    dy = ly / ny
    dz = lz / nz
    
    water_molecules = []
    count = 0
    
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if count >= n_water:
                    break
                
                x = (i + 0.5) * dx + np.random.uniform(-perturbation, perturbation)
                y = (j + 0.5) * dy + np.random.uniform(-perturbation, perturbation)
                z = z_start + (k + 0.5) * dz + np.random.uniform(-perturbation, perturbation)
                
                # Apply PBC
                x = x % lx
                y = y % ly
                z = max(z_start, min(z_end, z))
                
                pos = np.array([x, y, z])
                water = create_oriented_water(pos, orientation)
                water_molecules.append(water)
                count += 1
            if count >= n_water:
                break
        if count >= n_water:
            break
    
    return water_molecules


def place_water_interface(cell: np.ndarray,
                          n_water: int,
                          n_interface: int,
                          z_start: float,
                          interface_thickness: float,
                          water_layer_height: float,
                          interface_orientation: str = 'dangling',
                          bulk_orientation: str = 'random',
                          min_distance: float = 2.5,
                          seed: int = 42) -> List[Atoms]:
    """
    Place interface water layer with controlled orientation + bulk water above.
    
    Args:
        n_water: Total number of water molecules
        n_interface: Number of interface water molecules (first layer)
        interface_thickness: Thickness of interface layer (Å)
        interface_orientation: Orientation for interface waters
        bulk_orientation: Orientation for bulk waters
    """
    np.random.seed(seed)
    
    interface_z_end = z_start + interface_thickness
    bulk_z_start = interface_z_end
    bulk_z_end = z_start + water_layer_height
    
    n_bulk = n_water - n_interface
    
    print(f"  Interface layer: z={z_start:.2f} to {interface_z_end:.2f} Å, "
          f"{n_interface} waters ({interface_orientation})")
    print(f"  Bulk layer: z={bulk_z_start:.2f} to {bulk_z_end:.2f} Å, "
          f"{n_bulk} waters ({bulk_orientation})")
    
    # Place interface waters
    interface_waters = place_water_random(
        cell, n_interface, z_start, interface_z_end,
        min_distance, interface_orientation, seed
    )
    
    # Place bulk waters
    bulk_waters = place_water_random(
        cell, n_bulk, bulk_z_start, bulk_z_end,
        min_distance, bulk_orientation, seed + 1
    )
    
    return interface_waters + bulk_waters


def place_water_packmol(slab: Atoms,
                        n_water: int,
                        z_start: float,
                        z_end: float,
                        target_density: float = 1.0,
                        seed: int = 42) -> Atoms:
    """
    Use mdapackmol for density-controlled water packing.
    
    Returns combined structure with slab and water.
    """
    if not HAS_MDAPACKMOL:
        raise ImportError("mdapackmol is required for packmol mode. "
                          "Install with: pip install mdapackmol")
    
    cell = slab.get_cell()
    lx, ly = cell[0, 0], cell[1, 1]
    
    # Create temporary directory for PACKMOL files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write slab to PDB
        slab_pdb = os.path.join(tmpdir, 'slab.pdb')
        write(slab_pdb, slab, format='proteindatabank')
        
        # Write water molecule template
        water = molecule('H2O')
        water_pdb = os.path.join(tmpdir, 'water.pdb')
        write(water_pdb, water, format='proteindatabank')
        
        # Create PACKMOL structures
        slab_struct = PackmolStructure(
            slab_pdb,
            number=1,
            fixed=[0, 0, 0, 0, 0, 0]  # Fixed position
        )
        
        water_struct = PackmolStructure(
            water_pdb,
            number=n_water,
            inside_box=[0.5, 0.5, z_start, lx - 0.5, ly - 0.5, z_end]
        )
        
        # Run PACKMOL
        output_pdb = os.path.join(tmpdir, 'output.pdb')
        print(f"  Running PACKMOL to place {n_water} water molecules...")
        
        try:
            result = packmol([slab_struct, water_struct], 
                            output=output_pdb, 
                            seed=seed)
            
            # Read result
            combined = read(output_pdb, format='proteindatabank')
            combined.set_cell(cell)
            combined.set_pbc(slab.get_pbc())
            
            return combined
            
        except Exception as e:
            print(f"  PACKMOL failed: {e}")
            print("  Falling back to grid placement...")
            water_molecules = place_water_grid(
                cell, n_water, z_start, z_end, 'random', seed=seed
            )
            combined = slab.copy()
            for w in water_molecules:
                combined += w
            return combined


# ============================================================================
# Output Format Functions
# ============================================================================
def write_cp2k_format(atoms: Atoms, filename: str) -> None:
    """
    Write structure in CP2K compatible format (extended XYZ with cell info).
    
    CP2K can read extended XYZ format with proper headers.
    """
    cell = atoms.get_cell()
    
    with open(filename, 'w') as f:
        f.write(f"{len(atoms)}\n")
        
        # CP2K-compatible header with cell info
        a, b, c = cell[0], cell[1], cell[2]
        f.write(f'Lattice="{a[0]:.8f} {a[1]:.8f} {a[2]:.8f} ')
        f.write(f'{b[0]:.8f} {b[1]:.8f} {b[2]:.8f} ')
        f.write(f'{c[0]:.8f} {c[1]:.8f} {c[2]:.8f}" ')
        f.write('Properties=species:S:1:pos:R:3 ')
        pbc = atoms.get_pbc()
        f.write(f'pbc="{str(pbc[0])[0]} {str(pbc[1])[0]} {str(pbc[2])[0]}"\n')
        
        # Write atoms
        symbols = atoms.get_chemical_symbols()
        positions = atoms.get_positions()
        
        for sym, pos in zip(symbols, positions):
            f.write(f"{sym:2s} {pos[0]:16.8f} {pos[1]:16.8f} {pos[2]:16.8f}\n")
    
    print(f"  CP2K format saved to: {filename}")


def write_outputs(atoms: Atoms, base_path: str, formats: List[str]) -> None:
    """Write structure in multiple formats."""
    base = Path(base_path)
    stem = base.stem
    parent = base.parent
    
    for fmt in formats:
        if fmt == 'xyz':
            output_path = parent / f"{stem}.xyz"
            write(str(output_path), atoms, format='extxyz')
            print(f"  XYZ format saved to: {output_path}")
            
        elif fmt == 'cp2k':
            output_path = parent / f"{stem}_cp2k.xyz"
            write_cp2k_format(atoms, str(output_path))




# ============================================================================
# Main Building Function
# ============================================================================
def build_water_on_slab(
    slab: Atoms,
    n_water: int,
    placement_mode: str = 'random',
    z_offset: float = 2.0,
    water_layer_height: float = 15.0,
    min_water_distance: float = 2.5,
    target_density: float = 1.0,
    interface_layer_thickness: float = 5.0,
    interface_water_orientation: str = 'dangling',
    n_interface_water: Optional[int] = None,
    add_vacuum: bool = False,
    vacuum_thickness: float = 15.0,
    seed: int = 42
) -> Atoms:
    """
    Build water layer on top of slab structure.
    
    Args:
        slab: ASE Atoms object of the substrate
        n_water: Total number of water molecules
        placement_mode: 'random', 'grid', 'packmol', 'interface'
        z_offset: Distance above slab to start water layer
        water_layer_height: Height of water layer
        min_water_distance: Minimum O-O distance
        target_density: Target water density (g/cm³) for packmol mode
        interface_layer_thickness: Thickness of interface layer (Å)
        interface_water_orientation: Orientation for interface waters
        n_interface_water: Number of interface waters (auto if None)
        add_vacuum: Add vacuum gap above water
        vacuum_thickness: Vacuum gap thickness (Å)
        optimize_hbonds_flag: Optimize H-bond geometry
        seed: Random seed
    
    Returns:
        Combined Atoms object with slab and water
    """
    cell = slab.get_cell()
    positions = slab.get_positions()
    z_max_slab = np.max(positions[:, 2])
    
    z_start = z_max_slab + z_offset
    z_end = z_start + water_layer_height
    
    print(f"\n  Slab z_max: {z_max_slab:.2f} Å")
    print(f"  Water layer: z={z_start:.2f} to {z_end:.2f} Å")
    print(f"  Placement mode: {placement_mode}")
    
    # Place water molecules
    if placement_mode == 'packmol':
        combined = place_water_packmol(slab, n_water, z_start, z_end, 
                                        target_density, seed)
    else:
        if placement_mode == 'interface':
            if n_interface_water is None:
                # Auto-calculate based on surface area and 3.1 Å spacing
                lx, ly = cell[0, 0], cell[1, 1]
                area = lx * ly
                n_interface_water = int(area / (3.1 * 3.1))
                n_interface_water = min(n_interface_water, n_water // 2)
            
            water_molecules = place_water_interface(
                cell, n_water, n_interface_water, z_start,
                interface_layer_thickness, water_layer_height,
                interface_water_orientation, 'random',
                min_water_distance, seed
            )
        elif placement_mode == 'grid':
            water_molecules = place_water_grid(
                cell, n_water, z_start, z_end, 'random', 0.3, seed
            )
        else:  # random
            water_molecules = place_water_random(
                cell, n_water, z_start, z_end, 
                min_water_distance, 'random', seed
            )
        
        # Combine slab and water
        combined = slab.copy()
        for water in water_molecules:
            combined += water
    
    # Adjust cell height
    new_lz = z_end
    if add_vacuum:
        new_lz += vacuum_thickness
        print(f"  Adding vacuum gap: {vacuum_thickness:.2f} Å")
    else:
        new_lz += 5.0  # Buffer
    
    new_cell = cell.copy()
    new_cell[2, 2] = new_lz
    combined.set_cell(new_cell)
    
    # Wrap atoms into cell (ensure all atoms are inside box)
    combined.wrap()
    
    # H-bond optimization
    
    # Verify all atoms are inside box
    positions = combined.get_positions()
    cell_params = combined.get_cell()
    z_max_atom = np.max(positions[:, 2])
    if z_max_atom > new_lz:
        print(f"  Warning: Some atoms exceed box z-dimension ({z_max_atom:.2f} > {new_lz:.2f})")
    
    # Print summary
    n_water_actual = (len(combined) - len(slab)) // 3
    cell_lengths = combined.cell.cellpar()  # Updated to avoid deprecation warning
    a, b, c = cell_lengths[0], cell_lengths[1], cell_lengths[2]
    alpha, beta, gamma = cell_lengths[3], cell_lengths[4], cell_lengths[5]
    
    print(f"\n  Summary:")
    print(f"    Total atoms: {len(combined)}")
    print(f"    Slab atoms: {len(slab)}")
    print(f"    Water molecules: {n_water_actual}")
    print(f"    Cell: {a:.4f} {b:.4f} {c:.4f} {alpha:.2f} {beta:.2f} {gamma:.2f}")

    
    return combined


# ============================================================================
# Main Function
# ============================================================================
def main():
    """Main function to execute the workflow."""
    parser = argparse.ArgumentParser(
        description='Build water-PDMS interface for H2O2 mechanism studies.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
        # Basic random placement
        python build_water_on_silicone.py --n_water 100

        # For AIMD/DFT with interface layer and vacuum
        python build_water_on_silicone.py --n_water 30 --placement_mode interface `
            --interface_water_orientation dangling --add_vacuum --vacuum_thickness 15.0

        # High-quality PACKMOL placement
        python build_water_on_silicone.py --n_water 200 --placement_mode packmol `
            --target_density 1.0

        # Grid placement for large systems
        python build_water_on_silicone.py --n_water 500 --placement_mode grid
        """
    )
    
    # Basic parameters
    parser.add_argument('--n_water', type=int, default=100,
                        help='Number of water molecules (default: 100)')
    parser.add_argument('--placement_mode', type=str, default='interface',
                        choices=['random', 'grid', 'packmol', 'interface'],

                        help='Water placement mode (default: random)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    
    # Geometry parameters
    parser.add_argument('--z_offset', type=float, default=2.0,
                        help='Distance above slab to start placing water (default: 2.0 Å)')
    parser.add_argument('--water_layer_height', type=float, default=15.0,
                        help='Maximum height of water layer (default: 15.0 Å)')
    parser.add_argument('--min_water_distance', type=float, default=2.5,
                        help='Minimum O-O distance (default: 2.5 Å)')
    parser.add_argument('--slab_z_min', type=float, default=1.0,
                        help='Position slab bottom at this z coordinate (Å, default: 1.0)')

    
    # Interface layer parameters
    parser.add_argument('--interface_layer_thickness', type=float, default=5.0,
                        help='Thickness of interface water layer (default: 5.0 Å)')
    parser.add_argument('--interface_water_orientation', type=str, default='dangling',
                        choices=['dangling', 'parallel', 'random'],
                        help='Orientation of interface water (default: dangling)')
    parser.add_argument('--n_interface_water', type=int, default=None,
                        help='Number of interface water molecules (auto if not specified)')
    
    # Auto-calculation of interface parameters
    parser.add_argument('--auto_interface', action='store_true',
                        help='Auto-calculate interface layer parameters based on surface physics')
    parser.add_argument('--surface_type', type=str, default='hydrophobic',
                        choices=['hydrophobic', 'hydrophilic'],
                        help='Surface type for auto-calculation (default: hydrophobic)')
    parser.add_argument('--n_interface_layers', type=int, default=2,
                        help='Number of interface water layers: 1=monolayer, 2=bilayer (default: 2)')
    
    # Density control (for packmol)
    parser.add_argument('--target_density', type=float, default=1.0,
                        help='Target water density g/cm³ (default: 1.0)')
    
    # Vacuum gap
    parser.add_argument('--add_vacuum', action='store_true',
                        help='Add vacuum gap above water layer')
    parser.add_argument('--vacuum_thickness', type=float, default=30.0,
                        help='Vacuum gap thickness (default: 30.0 Å)')
    
    
    # Output
    parser.add_argument('--output_dir', type=str, default='output_model',
                        help='Output directory (default: output_model)')
    parser.add_argument('--output_formats', type=str, default='xyz,cp2k',
                        help='Output formats, comma-separated (default: xyz,cp2k)')
    parser.add_argument('--output_prefix', type=str, default=None,
                        help='Output file prefix (default: silicone_water_N)')
    parser.add_argument('--save_log', action='store_true', default=True,
                        help='Save run log to timestamped text file (default: True)')
    parser.add_argument('--no_log', action='store_true',
                        help='Disable log file saving')

    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Define file paths
    script_dir = Path(__file__).parent
    silicone_slab_file = script_dir / 'silicone_slab.xyz'
    densified_pdms_file = script_dir / 'densified_pdms.xyz'
    relaxed_structure_file = script_dir / 'relaxed_structure.xyz'
    
    # Check input files
    if not silicone_slab_file.exists():
        raise FileNotFoundError(f"silicone_slab.xyz not found at: {silicone_slab_file}")
    if not densified_pdms_file.exists():
        raise FileNotFoundError(f"densified_pdms.xyz not found at: {densified_pdms_file}")
    
    print("=" * 70)
    print("Water-PDMS Interface Builder for H2O2 Mechanism Studies")
    print("=" * 70)
    
    # Step 1: Create relaxed_structure.xyz
    print("\nStep 1: Creating relaxed_structure.xyz...")
    replace_header_and_save(str(silicone_slab_file), 
                            str(densified_pdms_file), 
                            str(relaxed_structure_file))
    
    # Step 2: Read slab structure
    print("\nStep 2: Reading slab structure...")
    slab = read(str(relaxed_structure_file))
    print(f"  Slab atoms: {len(slab)}")
    
    # Translate slab to position bottom at specified z coordinate
    if args.slab_z_min is not None:
        positions = slab.get_positions()
        current_z_min = np.min(positions[:, 2])
        z_shift = args.slab_z_min - current_z_min
        positions[:, 2] += z_shift
        slab.set_positions(positions)
        print(f"  Slab translated: z_min {current_z_min:.2f} -> {args.slab_z_min:.2f} A (shift: {z_shift:+.2f} A)")

    
    # Calculate interface parameters based on slab surface area
    cell = slab.get_cell()
    surface_area = cell[0, 0] * cell[1, 1]  # Å²
    
    # Auto-calculate interface parameters if requested
    if args.auto_interface:
        interface_params = calculate_interface_layer_params(
            surface_area=surface_area,
            surface_type=args.surface_type,
            n_layers=args.n_interface_layers,
            target_density=args.target_density
        )
        print_interface_params(interface_params)
        
        # Override defaults with calculated values
        interface_layer_thickness = interface_params['interface_thickness']
        z_offset = interface_params['z_offset']
        n_interface_water = interface_params['n_interface_max']
        
        # If using interface mode, cap n_interface_water at half of total water
        if args.placement_mode == 'interface':
            n_interface_water = min(n_interface_water, args.n_water // 2)
    else:
        interface_layer_thickness = args.interface_layer_thickness
        z_offset = args.z_offset
        n_interface_water = args.n_interface_water
    
    # Step 3: Add water molecules
    print(f"\nStep 3: Adding {args.n_water} water molecules...")
    combined = build_water_on_slab(
        slab=slab,
        n_water=args.n_water,
        placement_mode=args.placement_mode,
        z_offset=z_offset,
        water_layer_height=args.water_layer_height,
        min_water_distance=args.min_water_distance,
        target_density=args.target_density,
        interface_layer_thickness=interface_layer_thickness,
        interface_water_orientation=args.interface_water_orientation,
        n_interface_water=n_interface_water,
        add_vacuum=args.add_vacuum,
        vacuum_thickness=args.vacuum_thickness,
        seed=args.seed
    )

    
    # Step 4: Write output files
    print("\nStep 4: Writing output files...")
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"  Created output directory: {output_dir}")
    else:
        output_dir = script_dir
    
    # Determine output filename
    if args.output_prefix:
        output_base = output_dir / args.output_prefix
    else:
        output_base = output_dir / f'silicone_water_{args.n_water}'
    
    formats = [f.strip() for f in args.output_formats.split(',')]
    write_outputs(combined, str(output_base), formats)
    
    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)
    
    return combined, output_base, args

def run_with_logging():
    """Run main() with optional logging."""
    save_log = '--no_log' not in sys.argv
    
    if save_log:
        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        
        class Tee:
            def write(self, data):
                buffer.write(data)
                old_stdout.write(data)
            def flush(self):
                old_stdout.flush()
        sys.stdout = Tee()
    
    try:
        result, output_base, args = main()
    finally:
        if save_log:
            sys.stdout = old_stdout
    
    if save_log and not args.no_log:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f"{output_base}_{timestamp}.log"
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"# Run Log - {datetime.now():%Y-%m-%d %H:%M:%S}\n")
            f.write(f"# Command: {' '.join(sys.argv)}\n{'='*70}\n\n")
            f.write(buffer.getvalue())
        print(f"\n  Log saved to: {log_file}")


if __name__ == '__main__':
    run_with_logging()
