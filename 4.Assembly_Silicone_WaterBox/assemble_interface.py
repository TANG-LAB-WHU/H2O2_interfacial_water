#!/usr/bin/env python3
"""
This script reads a pre-equilibrated PDMS slab and a pre-equilibrated water box
directly from LAMMPS data files (.lmpdat) to leverage the exact image flags (nx, ny, nz).
This ensures BOTH the PDMS chains and the water molecules are perfectly unwrapped,
eliminating any "broken molecule" visual artifacts or simulation boundary explosion errors.

Usage:
    (1) no-field case without O2:
        python assemble_interface.py --z-gap 2.0 --vacuum-gap 30.0 --z-bottom 5.0 --use-pdms-xyz --use-water-xyz --no-wrap-water --output-dir no-field
    
    (2) charged interface without O2:
        python assemble_interface.py --z-gap 2.0 --vacuum-gap 30.0 --z-bottom 5.0 --edl-mode explicit --use-pdms-xyz --use-water-xyz --no-wrap-water --output-dir charged_interface
        
    (3) no-field with O2 sit in water slab:
        python assemble_interface.py --z-gap 2.0 --vacuum-gap 30.0 --z-bottom 5.0 --o2-mode water --use-pdms-xyz --use-water-xyz --no-wrap-water --output-dir no-field_o2_water
        
    (4) no-field with O2 close to PMDS surface:
        python assemble_interface.py --z-gap 2.0 --vacuum-gap 30.0 --z-bottom 5.0 --o2-mode interface --use-pdms-xyz --use-water-xyz --no-wrap-water --output-dir no-field_o2_interface
        
    (5) charged interface with O2 sit in water slab:
        python assemble_interface.py --z-gap 2.0 --vacuum-gap 30.0 --z-bottom 5.0 --edl-mode explicit --o2-mode water --use-pdms-xyz --use-water-xyz --no-wrap-water --output-dir charged_interface_o2_water
        
    (6) charged interface with O2 close to PMDS surface:
        python assemble_interface.py --z-gap 2.0 --vacuum-gap 30.0 --z-bottom 5.0 --edl-mode explicit --o2-mode interface --use-pdms-xyz --use-water-xyz --no-wrap-water --output-dir charged_interface_o2_interface
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
    Reads a LAMMPS data file, extracting masses to determine element symbols,
    and applies `nx, ny, nz` image flags to exactly unwrap coordinates.
    Sorts atoms by ID so that water molecules are sequentially grouped (O, H, H).
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
    mass_map = {}
    atoms_data = [] # List of tuples: (id, symbol, x, y, z)
    Lx = Ly = Lz = 0.0
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
                atom_id = int(parts[0])
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
                
                atoms_data.append((atom_id, mass_map[atom_type], x_unwrapped, y_unwrapped, z_unwrapped))
                
    if Lx == 0.0 or len(atoms_data) == 0:
        raise ValueError(f"Failed to parse data from {file_path}")
        
    # Sort atoms by ID to ensure molecules are grouped sequentially (O, H, H)
    atoms_data.sort(key=lambda item: item[0])
    
    symbols = [item[1] for item in atoms_data]
    positions = [[item[2], item[3], item[4]] for item in atoms_data]
        
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
    parser.add_argument('--no-center-xy', action='store_true',
                        help='Disable translating the entire system to center the PDMS mass in the X-Y box.')
    parser.add_argument('--no-shift-to-box', action='store_true',
                        help='Disable rigidly translating the entire system to positive X and Y coordinates.')
    parser.add_argument('--no-wrap-water', action='store_true',
                        help='Disable molecularly wrapping water molecules back into the primary X-Y cell.')
    
    # -------------------------------------------------------------------------
    # Contact Electrification & O2 Doping Modeling Parameters
    # -------------------------------------------------------------------------
    parser.add_argument('--edl-mode', choices=['none', 'explicit'], default='none',
                        help='Model control for Electric Double Layer. Default: none (neutral)')
    parser.add_argument('--o2-mode', choices=['none', 'interface', 'water'], default='none',
                        help='Model control for O2 molecule addition. Default: none')
    parser.add_argument('--edl-pairs', type=int, default=2,
                        help='Number of H3O+/OH- pairs to generate when --edl-mode is explicit. Default: 2')
    parser.add_argument('--o2-count', type=int, default=1,
                        help='Number of O2 molecules to add when --o2-mode is not none. Default: 1')
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--pdms', type=str, default=os.path.join(script_dir, "../2.Prerelax_Silicone_slab/lammps_melting-quenching_cuequ-NEWIMAGE-RTX5090/densified_pdms.lmpdat"), 
                        help='Path to PDMS .lmpdat file (used for cell dimensions and default unwrapped coordinates)')
    parser.add_argument('--water', type=str, default=os.path.join(script_dir, "../3.Prerelax_MatchingWater_Box/water_box_prerelaxation/water_box_relaxed.lmpdat"), 
                        help='Path to Water Box .lmpdat file (used for cell dimensions and default unwrapped coordinates)')
    parser.add_argument('--use-pdms-xyz', action='store_true',
                        help='Optional: Force using the densified_pdms.xyz file instead of the lmpdat file.')
    parser.add_argument('--use-water-xyz', action='store_true',
                        help='Optional: Force using the water_box_relaxed.xyz file instead of the lmpdat file.')
    parser.add_argument('--output-dir', type=str, default=script_dir, 
                        help='Directory to save the output file. Will be created if it does not exist.')
    parser.add_argument('--output', type=str, default="assembled_interface.xyz", 
                        help='Output .xyz filename')
                        
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
        
        # FIX: Liquid water unwrapping with `nz` scatters molecules across multiple 
        # Z-boxes due to diffusion. We MUST wrap them back molecularly in Z to 
        # restore the compact ~8.5 Å slab before assembly.
        water_Lx, water_Ly, water_Lz = water.get_cell().diagonal()
        pos = water.positions
        for i in range(0, len(water), 3):
            mol = pos[i:i+3]
            com_z = np.mean(mol[:, 2])
            shift_z = -np.floor(com_z / water_Lz) * water_Lz
            pos[i:i+3, 2] += shift_z
        water.set_positions(pos)
        print("    Molecularly wrapped diffused water molecules back into compact Z-slab.")
        
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
    
    # 4.5 Optional Visual Polishing (Centering and Wrapping - Enabled by default)
    if not args.no_shift_to_box:
        print(f"\n[3.5] Rigidly shifting entire system into the positive X-Y quadrant...")
        min_x = np.min(combined.positions[:, 0])
        min_y = np.min(combined.positions[:, 1])
        shift_x = -min_x + 0.5 
        shift_y = -min_y + 0.5
        combined.positions[:, 0] += shift_x
        combined.positions[:, 1] += shift_y
        print(f"    Shifted entire system by dx={shift_x:+.4f}, dy={shift_y:+.4f} A so min(X,Y) >= 0.")
        
    elif not args.no_center_xy:
        print(f"\n[3.5] Centering system in X-Y plane...")
        pdms_cx = (np.max(pdms.positions[:, 0]) + np.min(pdms.positions[:, 0])) / 2.0
        pdms_cy = (np.max(pdms.positions[:, 1]) + np.min(pdms.positions[:, 1])) / 2.0
        Lx, Ly = pdms.get_cell()[0, 0], pdms.get_cell()[1, 1]
        
        shift_x = (Lx / 2.0) - pdms_cx
        shift_y = (Ly / 2.0) - pdms_cy
        
        combined.positions[:, 0] += shift_x
        combined.positions[:, 1] += shift_y
        print(f"    Shifted entire system by dx={shift_x:+.4f}, dy={shift_y:+.4f} A to center PDMS.")
        
    if not args.no_wrap_water:
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

    # -------------------------------------------------------------------------
    # 4.8 Apply Physical and Chemical Modifications (EDL & O2)
    # -------------------------------------------------------------------------
    Lx, Ly = pdms.get_cell()[0, 0], pdms.get_cell()[1, 1]
    
    if args.edl_mode == 'explicit' or args.o2_mode != 'none':
        print(f"\n[4.8] Applying Physicochemical Modifications...")
        
        # Identify water molecules and compute true PDMS geometric center after shifting
        n_pdms = len(pdms)
        pos = combined.positions
        symbols = combined.get_chemical_symbols()
        
        pdms_positions = pos[:n_pdms]
        pdms_cx = (np.max(pdms_positions[:, 0]) + np.min(pdms_positions[:, 0])) / 2.0
        pdms_cy = (np.max(pdms_positions[:, 1]) + np.min(pdms_positions[:, 1])) / 2.0
        center_x, center_y = pdms_cx, pdms_cy
        
        water_molecules = [] 
        for i in range(n_pdms, len(combined), 3):
            if symbols[i] == 'O':
                ox, oy, oz = pos[i]
                dist_xy = np.sqrt((ox - center_x)**2 + (oy - center_y)**2)
                
                # Compute actual minimum 3D distance to PDMS atoms (incorporating lateral X-Y PBC)
                diffs = pdms_positions - pos[i]
                box_lengths = [Lx, Ly]
                for axis in range(2):
                    diffs[:, axis] = diffs[:, axis] - np.round(diffs[:, axis] / box_lengths[axis]) * box_lengths[axis]
                pdms_dists = np.linalg.norm(diffs, axis=1)
                min_dist_to_pdms = np.min(pdms_dists)
                
                water_molecules.append({
                    'o_idx': i,
                    'h1_idx': i+1,
                    'h2_idx': i+2,
                    'z': oz,
                    'dist_to_center': dist_xy,
                    'min_dist_to_pdms': min_dist_to_pdms
                })
        
        water_z_vals = [m['z'] for m in water_molecules]
        median_z = np.median(water_z_vals) if water_z_vals else 0.0

        used_water_indices = set()

        if args.edl_mode == 'explicit':
            print(f"    -> Building Explicit EDL ({args.edl_pairs} pairs)...")
            
            water_by_center = sorted(water_molecules, key=lambda m: m['dist_to_center'])
            central_waters = water_by_center[:max(30, args.edl_pairs * 10)]
            
            # Use the actual 3D minimum distance to PDMS atoms to identify interface waters
            central_by_dist = sorted(central_waters, key=lambda m: m['min_dist_to_pdms'])
            interface_waters = central_by_dist[:args.edl_pairs]
            
            central_by_bulk = sorted(central_waters, key=lambda m: abs(m['z'] - median_z))
            
            interface_indices = set(m['o_idx'] for m in interface_waters)
            bulk_waters = []
            for m in central_by_bulk:
                if m['o_idx'] not in interface_indices:
                    bulk_waters.append(m)
                if len(bulk_waters) == args.edl_pairs:
                    break
                    
            for iw, bw in zip(interface_waters, bulk_waters):
                h_idx = iw['h2_idx']
                bulk_o_pos = pos[bw['o_idx']]
                
                # Geometrically accurate H3O+ generation (Trigonal Pyramidal)
                h1_pos = pos[bw['h1_idx']]
                h2_pos = pos[bw['h2_idx']]
                vec_sum = (h1_pos - bulk_o_pos) + (h2_pos - bulk_o_pos)
                new_vec = -vec_sum
                new_vec = (new_vec / np.linalg.norm(new_vec)) * 1.0  # 1.0 A bond length
                pos[h_idx] = bulk_o_pos + new_vec
                print(f"      - Transferred proton {h_idx} from Z={iw['z']:.2f} (Interface OH-, d_PDMS={iw['min_dist_to_pdms']:.2f} A) to Z={bw['z']:.2f} (Bulk H3O+)")
                
                used_water_indices.add(iw['o_idx'])
                used_water_indices.add(bw['o_idx'])
            
            combined.set_positions(pos)
            
        if args.o2_mode == 'interface':
            print(f"    -> Adding O2 at Interface ({args.o2_count} molecules)...")
            o2_z = pdms_z_max + args.z_gap * 0.4
            o2_positions = []
            o2_symbols = []
            for i in range(args.o2_count):
                offset = i * 3.0 # safe vdW spacing for multiple O2
                o2_positions.append([center_x + offset, center_y, o2_z])
                o2_positions.append([center_x + offset, center_y, o2_z + 1.21])
                o2_symbols.extend(['O', 'O'])
                
            o2_atoms = Atoms(symbols=o2_symbols, positions=o2_positions)
            combined += o2_atoms
            print(f"      - Added {args.o2_count} O2 near geometric center Z={o2_z:.2f}")

        elif args.o2_mode == 'water':
            print(f"    -> Substituting Water with O2 in Bulk ({args.o2_count} molecules)...")
            
            water_by_center = sorted(water_molecules, key=lambda m: m['dist_to_center'])
            central_waters = water_by_center[:max(30, args.o2_count * 10)]
            central_by_bulk = sorted(central_waters, key=lambda m: abs(m['z'] - median_z))
            
            to_replace = []
            for m in central_by_bulk:
                if m['o_idx'] not in used_water_indices:
                    to_replace.append(m)
                if len(to_replace) == args.o2_count:
                    break
            
            indices_to_delete = []
            o2_positions = []
            o2_symbols = []
            for w in to_replace:
                indices_to_delete.extend([w['o_idx'], w['h1_idx'], w['h2_idx']])
                ox, oy, oz = pos[w['o_idx']]
                o2_positions.append([ox, oy, oz])
                o2_positions.append([ox, oy, oz + 1.21])
                o2_symbols.extend(['O', 'O'])
                print(f"      - Substituted water at Z={oz:.2f} with O2")
                
            del combined[np.array(indices_to_delete)]
            
            o2_atoms = Atoms(symbols=o2_symbols, positions=o2_positions)
            combined += o2_atoms

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
    final_output_path = os.path.join(args.output_dir, args.output)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
        
    print(f"\n[4] Writing final output to: {final_output_path}")
    # Write using ASE extended XYZ format which preserves lattice and PBC data natively for CP2K
    write(final_output_path, combined, format='extxyz')
    print("\nAssembly Completed Successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
