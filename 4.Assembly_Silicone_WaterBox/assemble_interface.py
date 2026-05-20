#!/usr/bin/env python3
"""
This script reads a pre-equilibrated PDMS slab and a pre-equilibrated water box
directly from LAMMPS data files (.lmpdat) to leverage the exact image flags (nx, ny, nz).
This ensures BOTH the PDMS chains and the water molecules are perfectly unwrapped,
eliminating any "broken molecule" visual artifacts or simulation boundary explosion errors.

Usage:
    python assemble_interface.py --z-gap 2.0 --vacuum-gap 30.0 --z-bottom 0.0
"""

import os
import argparse
import numpy as np
from ase.io import write
from ase import Atoms

def mass_to_symbol(mass: float) -> str:
    """Map atomic mass to chemical symbol."""
    m = round(mass)
    if m == 1: return 'H'
    if m == 12: return 'C'
    if m == 16: return 'O'
    if m == 28: return 'Si'
    raise ValueError(f"Unknown atomic mass {mass}")

def get_cell_from_lmpdat(file_path: str) -> np.ndarray:
    """Parses a LAMMPS data file to extract the orthogonal bounding box lengths."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    Lx = Ly = Lz = 0.0
    for line in lines:
        if 'xlo xhi' in line:
            Lx = float(line.split()[1]) - float(line.split()[0])
        elif 'ylo yhi' in line:
            Ly = float(line.split()[1]) - float(line.split()[0])
        elif 'zlo zhi' in line:
            Lz = float(line.split()[1]) - float(line.split()[0])
    if Lx == 0.0: raise ValueError(f"Failed to parse cell dimensions from {file_path}")
    return np.array([Lx, Ly, Lz])

def unwrap_water_box(water_atoms: Atoms) -> Atoms:
    """Heuristic unwrap for water molecules from XYZ without image flags."""
    positions = water_atoms.get_positions()
    symbols = water_atoms.get_chemical_symbols()
    box_lengths = water_atoms.get_cell().lengths()
    unwrapped_pos = positions.copy()
    
    for i in range(0, len(water_atoms), 3):
        if symbols[i] != 'O': continue
        o_pos = positions[i]
        for h_idx in (i+1, i+2):
            d = positions[h_idx] - o_pos
            for axis in range(3):
                L = box_lengths[axis]
                if d[axis] > L / 2.0: unwrapped_pos[h_idx, axis] -= L
                elif d[axis] < -L / 2.0: unwrapped_pos[h_idx, axis] += L
    
    unwrapped = water_atoms.copy()
    unwrapped.set_positions(unwrapped_pos)
    return unwrapped

def read_lmpdat_unwrapped(file_path: str) -> Atoms:
    """
    Parses a LAMMPS data file, extracting the exact unwrapped coordinates
    using the image flags (nx, ny, nz) generated during the MD simulation.
    Returns an ASE Atoms object.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
    Lx = Ly = Lz = 0.0
    mass_map = {} # type -> symbol
    
    positions = []
    symbols = []
    
    mode = None
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue
            
        if 'xlo xhi' in line:
            parts = line.split()
            Lx = float(parts[1]) - float(parts[0])
        elif 'ylo yhi' in line:
            parts = line.split()
            Ly = float(parts[1]) - float(parts[0])
        elif 'zlo zhi' in line:
            parts = line.split()
            Lz = float(parts[1]) - float(parts[0])
        elif stripped == 'Masses':
            mode = 'masses'
            continue
        elif stripped.startswith('Atoms'):
            mode = 'atoms'
            continue
        elif stripped == 'Velocities' or stripped == 'Bonds':
            mode = None
            
        if mode == 'masses':
            parts = stripped.split()
            if len(parts) >= 2:
                atom_type = int(parts[0])
                mass = float(parts[1])
                mass_map[atom_type] = mass_to_symbol(mass)
                
        elif mode == 'atoms':
            parts = stripped.split()
            if len(parts) >= 8:
                # Format: id type x y z nx ny nz
                atom_type = int(parts[1])
                x = float(parts[2])
                y = float(parts[3])
                z = float(parts[4])
                nx = int(parts[5])
                ny = int(parts[6])
                nz = int(parts[7])
                
                # Apply exact unwrapping using periodic box lengths
                x_unwrapped = x + nx * Lx
                y_unwrapped = y + ny * Ly
                z_unwrapped = z + nz * Lz
                
                positions.append([x_unwrapped, y_unwrapped, z_unwrapped])
                symbols.append(mass_map[atom_type])
                
    if Lx == 0.0 or len(positions) == 0:
        raise ValueError(f"Failed to parse data from {file_path}")
        
    atoms = Atoms(symbols=symbols, positions=positions, cell=[Lx, Ly, Lz], pbc=[True, True, True])
    return atoms

def main():
    parser = argparse.ArgumentParser(description="Assemble PDMS and Water Box interfaces.")
    parser.add_argument('--z-gap', type=float, default=2.0, 
                        help='Van der Waals gap between PDMS and water layer (Angstroms). Default: 2.0')
    parser.add_argument('--vacuum-gap', type=float, default=30.0, 
                        help='Vacuum layer thickness above the water (Angstroms). Default: 30.0')
    parser.add_argument('--z-bottom', type=float, default=1.0, 
                        help='Shift the entire system so the bottom-most atom sits at this Z coordinate. Default: 1.0')
    parser.add_argument('--center-xy', action='store_true',
                        help='Translate the entire system so the PDMS center of mass is centered in the X-Y periodic box.')
    parser.add_argument('--shift-to-box', action='store_true',
                        help='Rigidly translate the entire system so that all X and Y coordinates are >= 0 (moves atoms into the VMD box without wrapping).')
    parser.add_argument('--wrap-water', action='store_true',
                        help='Molecularly wrap water molecules back into the primary X-Y cell (useful for clean visualization).')
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--pdms', type=str, default=os.path.join(script_dir, "../2.Prerelax_Silicone_slab/lammps_melting-quenching_cuequ-NEWIMAGE-RTX5090/densified_pdms.lmpdat"), 
                        help='Path to PDMS .lmpdat file (used for cell dimensions and default unwrapped coordinates)')
    parser.add_argument('--water', type=str, default=os.path.join(script_dir, "../3.Prerelax_MatchingWater_Box/water_box_prerelaxation/water_box_relaxed.lmpdat"), 
                        help='Path to Water Box .lmpdat file (used for cell dimensions and default unwrapped coordinates)')
    parser.add_argument('--use-pdms-xyz', action='store_true',
                        help='Optional: Force using the densified_pdms.xyz file instead of the lmpdat file.')
    parser.add_argument('--use-water-xyz', action='store_true',
                        help='Optional: Force using the water_box_relaxed.xyz file instead of the lmpdat file.')
    parser.add_argument('--output', type=str, default=os.path.join(script_dir, "assembled_interface.xyz"), 
                        help='Output .xyz file path')
                        
    args = parser.parse_args()

    print("=" * 60)
    print("  Advanced PDMS-Water Interface Assembly Builder")
    print("=" * 60)
    
    if not os.path.exists(args.pdms) or not os.path.exists(args.water):
        raise FileNotFoundError(f"Input lmpdat files not found. Please check paths:\nPDMS: {args.pdms}\nWater: {args.water}")

    # 1. Read structures
    print(f"\n[1] Reading structures...")
    
    # Process PDMS
    pdms_xyz_path = os.path.join(script_dir, "../2.Prerelax_Silicone_slab/lammps_melting-quenching_cuequ-NEWIMAGE-RTX5090/densified_pdms.xyz")
    if args.use_pdms_xyz and os.path.exists(pdms_xyz_path):
        from ase.io import read
        print(f"    Reading PDMS from XYZ: {pdms_xyz_path}")
        pdms = read(pdms_xyz_path)
        pdms.set_cell(get_cell_from_lmpdat(args.pdms))
        pdms.set_pbc([True, True, True])
        print("    Warning: Reading from XYZ loses LAMMPS image flags. PDMS may show broken bonds across PBC.")
    else:
        print(f"    Reading and exactly unwrapping PDMS from lmpdat...")
        pdms = read_lmpdat_unwrapped(args.pdms)
        
    # Process Water
    water_xyz_path = os.path.join(script_dir, "../3.Prerelax_MatchingWater_Box/water_box_prerelaxation/water_box_relaxed.xyz")
    if args.use_water_xyz and os.path.exists(water_xyz_path):
        from ase.io import read
        print(f"    Reading Water from XYZ: {water_xyz_path}")
        water = read(water_xyz_path)
        water.set_cell(get_cell_from_lmpdat(args.water))
        water.set_pbc([True, True, True])
        print("    Directly using raw coordinates from XYZ (no unwrapping applied).")
    else:
        print(f"    Reading and exactly unwrapping Water from lmpdat...")
        water = read_lmpdat_unwrapped(args.water)
    
    print(f"    PDMS atoms: {len(pdms)}")
    print(f"    Water atoms: {len(water)} ({len(water)//3} molecules)")
    
    # 2. Align the origin (z-bottom)
    print(f"\n[2] Aligning structures...")
    pdms_z_min = np.min(pdms.positions[:, 2])
    # Shift PDMS so its bottom sits exactly at args.z_bottom
    shift_pdms_z = args.z_bottom - pdms_z_min
    pdms.positions[:, 2] += shift_pdms_z
    print(f"    Shifted PDMS by {shift_pdms_z:+.4f} A to rest at Z = {args.z_bottom} A.")
    
    # 3. Stack the water box
    pdms_z_max = np.max(pdms.positions[:, 2])
    water_z_min = np.min(water.positions[:, 2])
    
    # Calculate how much to shift the water so it sits exactly at pdms_z_max + args.z_gap
    target_water_bottom = pdms_z_max + args.z_gap
    shift_water_z = target_water_bottom - water_z_min
    
    water.positions[:, 2] += shift_water_z
    print(f"    PDMS Z-max surface is now at: {pdms_z_max:.4f} A")
    print(f"    Shifted Water Box by {shift_water_z:+.4f} A to start at Z = {target_water_bottom:.4f} A.")
    
    # 4. Combine systems
    combined = pdms.copy()
    combined += water
    
    # 4.5 Optional Visual Polishing (Centering and Wrapping)
    if args.shift_to_box:
        print(f"\n[3.5] Rigidly shifting entire system into the positive X-Y quadrant...")
        min_x = np.min(combined.positions[:, 0])
        min_y = np.min(combined.positions[:, 1])
        shift_x = -min_x + 0.5 
        shift_y = -min_y + 0.5
        combined.positions[:, 0] += shift_x
        combined.positions[:, 1] += shift_y
        print(f"    Shifted entire system by dx={shift_x:+.4f}, dy={shift_y:+.4f} A so min(X,Y) >= 0.")
        
    elif args.center_xy:
        print(f"\n[3.5] Centering system in X-Y plane...")
        pdms_cx = (np.max(pdms.positions[:, 0]) + np.min(pdms.positions[:, 0])) / 2.0
        pdms_cy = (np.max(pdms.positions[:, 1]) + np.min(pdms.positions[:, 1])) / 2.0
        Lx, Ly = pdms.get_cell()[0, 0], pdms.get_cell()[1, 1]
        
        shift_x = (Lx / 2.0) - pdms_cx
        shift_y = (Ly / 2.0) - pdms_cy
        
        combined.positions[:, 0] += shift_x
        combined.positions[:, 1] += shift_y
        print(f"    Shifted entire system by dx={shift_x:+.4f}, dy={shift_y:+.4f} A")
        
    if args.wrap_water:
        print(f"    Molecularly wrapping water molecules back into primary cell...")
        Lx, Ly = pdms.get_cell()[0, 0], pdms.get_cell()[1, 1]
        n_pdms = len(pdms)
        pos = combined.positions
        for i in range(n_pdms, len(combined), 3):
            # Calculate Center of Mass for the water molecule
            mol = pos[i:i+3]
            com_x, com_y = np.mean(mol[:, 0]), np.mean(mol[:, 1])
            
            # Find the periodic shift needed to put COM into [0, L]
            shift_x = -np.floor(com_x / Lx) * Lx
            shift_y = -np.floor(com_y / Ly) * Ly
            
            pos[i:i+3, 0] += shift_x
            pos[i:i+3, 1] += shift_y
        combined.set_positions(pos)

    # 5. Rebuild periodic cell for downstream AIMD (CP2K)
    print(f"\n[4] Rebuilding periodic cell...")
    combined_z_max = np.max(combined.get_positions()[:, 2])
    new_lz = combined_z_max + args.vacuum_gap
    
    # Inherit X and Y lateral dimensions from the perfectly matched PDMS cell
    new_cell = pdms.get_cell().copy()
    new_cell[2, 2] = new_lz
    combined.set_cell(new_cell)
    
    print(f"    New Cell dimensions:")
    print(f"    Lx = {new_cell[0,0]:.4f} A")
    print(f"    Ly = {new_cell[1,1]:.4f} A")
    print(f"    Lz = {new_cell[2,2]:.4f} A (Includes {args.vacuum_gap} A vacuum)")
    
    # 6. Export the final assembled structure
    print(f"\n[4] Writing final output to: {args.output}")
    # Write using ASE extended XYZ format which preserves lattice and PBC data natively for CP2K
    write(args.output, combined, format='extxyz')
    print("\nAssembly Completed Successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
