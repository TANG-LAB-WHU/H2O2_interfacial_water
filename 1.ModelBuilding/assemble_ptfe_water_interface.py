#!/usr/bin/env python3
"""
Assemble PTFE slab and water droplet into a liquid-solid contact model.
PTFE slab is positioned at the bottom, water droplet above with controllable gap.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime


def read_xyz(filename):
    """Read XYZ file and return atoms list and comment line."""
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    if len(lines) < 2:
        raise ValueError(f"Invalid XYZ file: {filename}")
    
    try:
        num_atoms = int(lines[0].strip())
    except ValueError:
        raise ValueError(f"Invalid number of atoms in first line: {filename}")
    
    comment = lines[1].strip()
    
    atoms = []
    for i in range(2, 2 + num_atoms):
        if i >= len(lines):
            raise ValueError(f"Not enough atoms in file: {filename}")
        
        parts = lines[i].strip().split()
        if len(parts) < 4:
            continue
        
        symbol = parts[0]
        try:
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            atoms.append((symbol, x, y, z))
        except ValueError:
            continue
    
    if len(atoms) != num_atoms:
        print(f"Warning: Expected {num_atoms} atoms, found {len(atoms)}", file=sys.stderr)
    
    return atoms, comment


def write_xyz(filename, atoms, comment, box_info=None):
    """Write atoms to XYZ file with optional box and lattice parameter information.
    
    Args:
        filename: Output filename
        atoms: List of (symbol, x, y, z) tuples
        comment: Comment line (will include box info if provided)
        box_info: Tuple of (lx, ly, lz, (x_min, x_max), (y_min, y_max), (z_min, z_max), (a, b, c, alpha, beta, gamma)) or None
    """
    with open(filename, 'w') as f:
        f.write(f"{len(atoms)}\n")
        
        # Add box and lattice parameter information to comment if provided
        if box_info:
            lx, ly, lz, (x_min, x_max), (y_min, y_max), (z_min, z_max), (a, b, c, alpha, beta, gamma) = box_info
            comment_with_box = (f"{comment} | Box: Lx={lx:.6f} Ly={ly:.6f} Lz={lz:.6f} | "
                              f"X:[{x_min:.6f},{x_max:.6f}] Y:[{y_min:.6f},{y_max:.6f}] Z:[{z_min:.6f},{z_max:.6f}] | "
                              f"Lattice: {a:.6f} {b:.6f} {c:.6f} {alpha:.2f} {beta:.2f} {gamma:.2f}")
        else:
            comment_with_box = comment
        
        f.write(f"{comment_with_box}\n")
        for symbol, x, y, z in atoms:
            f.write(f"{symbol:3s} {x:15.9f} {y:15.9f} {z:15.9f}\n")


def get_z_range(atoms):
    """Get minimum and maximum z-coordinates."""
    if not atoms:
        return 0.0, 0.0
    
    z_coords = [z for _, _, _, z in atoms]
    return min(z_coords), max(z_coords)


def get_box_dimensions(atoms):
    """Get box dimensions and lattice parameters for the system.
    
    Returns:
        Tuple of (lx, ly, lz, (x_min, x_max), (y_min, y_max), (z_min, z_max), (a, b, c, alpha, beta, gamma))
        For orthogonal box: a=lx, b=ly, c=lz, alpha=beta=gamma=90.0
    """
    if not atoms:
        return 0.0, 0.0, 0.0, (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0, 0.0, 90.0, 90.0, 90.0)
    
    x_coords = [x for _, x, _, _ in atoms]
    y_coords = [y for _, _, y, _ in atoms]
    z_coords = [z for _, _, _, z in atoms]
    
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    z_min, z_max = min(z_coords), max(z_coords)
    
    lx = x_max - x_min
    ly = y_max - y_min
    lz = z_max - z_min
    
    # For orthogonal box, lattice parameters equal box dimensions
    # alpha, beta, gamma = 90.0 degrees for orthogonal box
    a, b, c = lx, ly, lz
    alpha, beta, gamma = 90.0, 90.0, 90.0
    
    return lx, ly, lz, (x_min, x_max), (y_min, y_max), (z_min, z_max), (a, b, c, alpha, beta, gamma)


def shift_atoms(atoms, dx, dy, dz):
    """Shift all atoms by given offsets."""
    return [(symbol, x + dx, y + dy, z + dz) for symbol, x, y, z in atoms]


def wrap_atoms(atoms, x_min, x_max, y_min, y_max, z_min, z_max):
    """Wrap atoms back into the box if they are outside the boundaries.
    
    This applies periodic boundary conditions to wrap atoms that are outside
    the box boundaries back into the box [x_min, x_max] x [y_min, y_max] x [z_min, z_max].
    
    Args:
        atoms: List of (symbol, x, y, z) tuples
        x_min, x_max: X boundaries
        y_min, y_max: Y boundaries
        z_min, z_max: Z boundaries
    
    Returns:
        List of wrapped atoms with coordinates within the box boundaries
    """
    lx = x_max - x_min
    ly = y_max - y_min
    lz = z_max - z_min
    
    wrapped_atoms = []
    for symbol, x, y, z in atoms:
        # Wrap X coordinate
        while x < x_min:
            x += lx
        while x >= x_max:
            x -= lx
        
        # Wrap Y coordinate
        while y < y_min:
            y += ly
        while y >= y_max:
            y -= ly
        
        # Wrap Z coordinate
        while z < z_min:
            z += lz
        while z >= z_max:
            z -= lz
        
        wrapped_atoms.append((symbol, x, y, z))
    
    return wrapped_atoms


def remove_overlapping_water_molecules(water_atoms, ptfe_z_min, ptfe_z_max):
    """Remove water molecules that intersect the PTFE slab or lie below its bottom.
    
    A water molecule is removed if:
      - any of its atoms has z-coordinate within the PTFE slab z-range
        [ptfe_z_min, ptfe_z_max]  (intersecting the slab), OR
      - all of its atoms have z < ptfe_z_min (molecule completely below the slab).
    
    Args:
        water_atoms: List of (symbol, x, y, z) tuples for water atoms
        ptfe_z_min: Minimum z-coordinate of PTFE slab
        ptfe_z_max: Maximum z-coordinate of PTFE slab
    
    Returns:
        Filtered list of water atoms with overlapping molecules removed
    """
    if not water_atoms:
        return water_atoms
    
    # Group atoms by molecule (assuming water molecules are consecutive: O, H, H)
    # This is a simple approach - for more complex cases, might need distance-based grouping
    water_molecules = []
    current_molecule = []
    
    for atom in water_atoms:
        symbol, x, y, z = atom
        if symbol == 'O':
            # Start a new molecule
            if current_molecule:
                water_molecules.append(current_molecule)
            current_molecule = [atom]
        else:
            # H atom, add to current molecule
            current_molecule.append(atom)
    
    # Add the last molecule
    if current_molecule:
        water_molecules.append(current_molecule)
    
    # Filter out molecules that overlap with PTFE slab or lie completely below it
    filtered_molecules = []
    removed_count = 0
    
    for molecule in water_molecules:
        z_values = [z for _, _, _, z in molecule]

        # Condition 1: any atom intersects the slab
        intersects_slab = any(ptfe_z_min <= z <= ptfe_z_max for z in z_values)

        # Condition 2: molecule lies entirely below the slab bottom
        completely_below_slab = max(z_values) < ptfe_z_min

        if not (intersects_slab or completely_below_slab):
            filtered_molecules.append(molecule)
        else:
            removed_count += 1
    
    # Flatten the filtered molecules back to atom list
    filtered_atoms = []
    for molecule in filtered_molecules:
        filtered_atoms.extend(molecule)
    
    if removed_count > 0:
        print(f"  Removed {removed_count} water molecules overlapping with PTFE slab")
    
    return filtered_atoms


class Tee:
    """A class that writes to both stdout and a file simultaneously."""
    def __init__(self, file_path):
        self.file = open(file_path, 'w', encoding='utf-8')
        self.stdout = sys.stdout
    
    def write(self, text):
        self.stdout.write(text)
        self.file.write(text)
        self.file.flush()
    
    def flush(self):
        self.stdout.flush()
        self.file.flush()
    
    def close(self):
        if self.file:
            self.file.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Assemble PTFE slab and water droplet into liquid-solid interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Assemble with default PTFE file, 3.0 Angstrom gap (water droplet centered by default)
  python assemble_ptfe_water_interface.py --water-file water_microdroplets/H2O_droplet_8angstroms.xyz -g 3.0

  # Assemble with custom PTFE file and 5.0 Angstrom gap
  python assemble_ptfe_water_interface.py --ptfe-file custom_ptfe.xyz --water-file water_microdroplets/H2O_droplet_10angstroms.xyz -g 5.0 -o interface.xyz

  # Assemble with output directory and no centering
  python assemble_ptfe_water_interface.py --water-file water_microdroplets/H2O_droplet_8angstroms.xyz -g 3.0 --output-dir output/ --no-center-xy
        """
    )
    
    parser.add_argument(
        '--ptfe-file',
        type=str,
        default='step1-ptfe_slab/manual_packmol/ptfe_bulk.xyz',
        help='Path to PTFE bulk XYZ file (default: step1-ptfe_slab/manual_packmol/ptfe_bulk.xyz)'
    )
    
    parser.add_argument(
        '--water-file',
        type=str,
        required=True,
        help='Path to water droplet XYZ file'
    )
    
    parser.add_argument(
        '-g', '--gap',
        type=float,
        default=3.0,
        help='Gap between PTFE slab top and water droplet bottom (Angstroms, default: 3.0)'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='ptfe_water_interface.xyz',
        help='Output XYZ filename (default: ptfe_water_interface.xyz)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for the output file (default: current directory)'
    )
    
    parser.add_argument(
        '--no-center-xy',
        dest='center_xy',
        action='store_false',
        default=True,
        help='Disable centering of water droplet in XY plane relative to PTFE slab (default: centered)'
    )
    
    parser.add_argument(
        '--vacuum-height',
        type=float,
        default=10.0,
        help='Vacuum layer height above the system (Angstroms, default: 10.0)'
    )
    
    parser.add_argument(
        '--bottom-offset',
        type=float,
        default=0.0,
        help='Distance from origin to bottom of the system (Angstroms, default: 0.0)'
    )
    
    args = parser.parse_args()
    
    # Setup logging to file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"assemble_ptfe_water_interface_{timestamp}.log"
    
    # Determine log file directory (use output_dir if specified, otherwise current directory)
    if args.output_dir:
        log_dir = Path(args.output_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / log_filename
    else:
        log_path = Path(log_filename)
    
    # Redirect all output (both stdout and stderr) to log file while also printing to console
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    tee_stdout = Tee(str(log_path))
    sys.stdout = tee_stdout
    sys.stderr = tee_stdout
    
    print(f"Log file: {log_path}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print()
    
    try:
        # Read PTFE slab
        print(f"Reading PTFE slab from: {args.ptfe_file}")
        try:
            ptfe_atoms, ptfe_comment = read_xyz(args.ptfe_file)
        except Exception as e:
            print(f"Error reading PTFE file: {e}", file=sys.stderr)
            sys.exit(1)
        
        print(f"  Found {len(ptfe_atoms)} atoms in PTFE slab")
        
        # Read water droplet
        print(f"Reading water droplet from: {args.water_file}")
        try:
            water_atoms, water_comment = read_xyz(args.water_file)
        except Exception as e:
            print(f"Error reading water file: {e}", file=sys.stderr)
            sys.exit(1)
        
        print(f"  Found {len(water_atoms)} atoms in water droplet")
        
        # Get z-ranges
        ptfe_z_min, ptfe_z_max = get_z_range(ptfe_atoms)
        water_z_min, water_z_max = get_z_range(water_atoms)
        
        print(f"\nPTFE slab z-range: {ptfe_z_min:.3f} to {ptfe_z_max:.3f} Angstroms")
        print(f"Water droplet z-range: {water_z_min:.3f} to {water_z_max:.3f} Angstroms")
        
        # Calculate shift for water droplet
        # Position PTFE so its minimum z is at 0 (optional, but keeps coordinates reasonable)
        # Then place water droplet so its minimum z is at (ptfe_z_max + gap)
        water_z_shift = ptfe_z_max + args.gap - water_z_min
        
        # Calculate XY shifts - default is to center water droplet on PTFE slab
        if args.center_xy:
            # Get XY ranges for PTFE
            ptfe_x_coords = [x for _, x, _, _ in ptfe_atoms]
            ptfe_y_coords = [y for _, _, y, _ in ptfe_atoms]
            ptfe_x_center = (min(ptfe_x_coords) + max(ptfe_x_coords)) / 2.0
            ptfe_y_center = (min(ptfe_y_coords) + max(ptfe_y_coords)) / 2.0
            
            # Get XY ranges for water
            water_x_coords = [x for _, x, _, _ in water_atoms]
            water_y_coords = [y for _, _, y, _ in water_atoms]
            water_x_center = (min(water_x_coords) + max(water_x_coords)) / 2.0
            water_y_center = (min(water_y_coords) + max(water_y_coords)) / 2.0
            
            water_x_shift = ptfe_x_center - water_x_center
            water_y_shift = ptfe_y_center - water_y_center
            
            print(f"\nCentering water droplet on PTFE slab:")
            print(f"  PTFE center: ({ptfe_x_center:.3f}, {ptfe_y_center:.3f}) Angstroms")
            print(f"  Water center (before shift): ({water_x_center:.3f}, {water_y_center:.3f}) Angstroms")
        else:
            water_x_shift = 0.0
            water_y_shift = 0.0
            print(f"\nWater droplet XY position unchanged (not centered)")
        
        # Shift water droplet
        water_atoms_shifted = shift_atoms(water_atoms, water_x_shift, water_y_shift, water_z_shift)
        
        # Remove overlapping water molecules if gap < 0
        if args.gap < 0:
            print(f"\nGap is negative ({args.gap:.3f} A), removing overlapping water molecules...")
            water_atoms_shifted = remove_overlapping_water_molecules(water_atoms_shifted, ptfe_z_min, ptfe_z_max)
        
        # Get final z-range for water
        water_z_min_final, water_z_max_final = get_z_range(water_atoms_shifted)
        
        print(f"\nWater droplet positioned:")
        print(f"  XY shift: ({water_x_shift:.3f}, {water_y_shift:.3f}) Angstroms")
        print(f"  Z shift: {water_z_shift:.3f} Angstroms")
        print(f"  Final z-range: {water_z_min_final:.3f} to {water_z_max_final:.3f} Angstroms")
        print(f"  Gap between PTFE top and water bottom: {water_z_min_final - ptfe_z_max:.3f} Angstroms")
        
        # Update water atom count after potential removal
        num_water_atoms_final = len(water_atoms_shifted)
        if args.gap < 0 and num_water_atoms_final != len(water_atoms):
            print(f"  Water atoms after removal: {num_water_atoms_final} (original: {len(water_atoms)})")
        
        # Combine atoms
        combined_atoms = ptfe_atoms + water_atoms_shifted
        
        # ------------------------------------------------------------------
        # Shift system so that:
        #   - Bottom of all atoms is at z = bottom_offset
        #   - Box in X and Y starts at 0 (i.e. x_min = 0, y_min = 0)
        # This makes the written box [0, Lx] × [0, Ly] × [bottom_offset, bottom_offset + Lz]
        # and avoids atoms appearing outside the visual cell in viewers.
        # ------------------------------------------------------------------
        # 1) Z shift -> bottom_offset
        current_z_min = min(z for _, _, _, z in combined_atoms)
        z_shift_to_origin = args.bottom_offset - current_z_min
        combined_atoms = shift_atoms(combined_atoms, 0.0, 0.0, z_shift_to_origin)
        print(f"\nShifting system to set bottom offset:")
        print(f"  Bottom offset from origin: {args.bottom_offset:.3f} Angstroms")
        print(f"  Original z_min: {current_z_min:.3f} Angstroms")
        print(f"  Z shift applied: {z_shift_to_origin:.3f} Angstroms")
        
        # 2) Shift X/Y so that x_min = 0, y_min = 0
        lx, ly, lz_atoms, (x_min, x_max), (y_min, y_max), (z_min, z_max), (a_atoms, b_atoms, c_atoms, alpha, beta, gamma) = get_box_dimensions(combined_atoms)
        x_shift = -x_min
        y_shift = -y_min
        if x_shift != 0.0 or y_shift != 0.0:
            combined_atoms = shift_atoms(combined_atoms, x_shift, y_shift, 0.0)
            print(f"\nShifting system in XY to align box with origin:")
            print(f"  X shift applied: {x_shift:.3f} Angstroms")
            print(f"  Y shift applied: {y_shift:.3f} Angstroms")
            # Recompute dimensions after XY shift
            lx, ly, lz_atoms, (x_min, x_max), (y_min, y_max), (z_min, z_max), (a_atoms, b_atoms, c_atoms, alpha, beta, gamma) = get_box_dimensions(combined_atoms)
        
        # Add vacuum layer to z-direction
        lz = lz_atoms + args.vacuum_height
        a = lx  # Lattice parameter a equals Lx
        b = ly  # Lattice parameter b equals Ly
        c = c_atoms + args.vacuum_height  # Lattice parameter c includes vacuum
        
        # Update z_max to include vacuum (box top)
        z_max_with_vacuum = z_min + lz
        
        print(f"\nBox dimensions (before vacuum):")
        print(f"  Lx = {lx:.6f} Angstroms (X: {x_min:.6f} to {x_max:.6f})")
        print(f"  Ly = {ly:.6f} Angstroms (Y: {y_min:.6f} to {y_max:.6f})")
        print(f"  Lz (atoms only) = {lz_atoms:.6f} Angstroms (Z: {z_min:.6f} to {z_max:.6f})")
        
        print(f"\nBox dimensions (with vacuum):")
        print(f"  Lx = {lx:.6f} Angstroms")
        print(f"  Ly = {ly:.6f} Angstroms")
        print(f"  Lz = {lz:.6f} Angstroms (Z: {z_min:.6f} to {z_max_with_vacuum:.6f}, vacuum: {args.vacuum_height:.3f} A)")
        
        # Verify and wrap atoms to ensure they are within the box
        x_coords = [x for _, x, _, _ in combined_atoms]
        y_coords = [y for _, _, y, _ in combined_atoms]
        z_coords = [z for _, _, _, z in combined_atoms]
        
        atoms_outside = []
        for i, (symbol, x, y, z) in enumerate(combined_atoms):
            if x < x_min or x >= x_max or y < y_min or y >= y_max or z < z_min or z >= z_max_with_vacuum:
                atoms_outside.append((i, symbol, x, y, z))
        
        if atoms_outside:
            print(f"\nWarning: {len(atoms_outside)} atoms are outside the box boundaries!")
            for i, symbol, x, y, z in atoms_outside[:10]:  # Show first 10
                print(f"  Atom {i}: {symbol} at ({x:.3f}, {y:.3f}, {z:.3f})")
            if len(atoms_outside) > 10:
                print(f"  ... and {len(atoms_outside) - 10} more")
            print(f"Wrapping atoms back into the box...")
            combined_atoms = wrap_atoms(combined_atoms, x_min, x_max, y_min, y_max, z_min, z_max_with_vacuum)
            print(f"  Wrapped {len(atoms_outside)} atoms back into the box boundaries.")
            
            # Verify all atoms are now within the box after wrapping
            atoms_outside_after = []
            for i, (symbol, x, y, z) in enumerate(combined_atoms):
                if x < x_min or x >= x_max or y < y_min or y >= y_max or z < z_min or z >= z_max_with_vacuum:
                    atoms_outside_after.append((i, symbol, x, y, z))
            if atoms_outside_after:
                print(f"  Warning: {len(atoms_outside_after)} atoms are still outside after wrapping!")
            else:
                print(f"  All atoms are now within the box boundaries.")
        else:
            print(f"\nAll atoms are within the box boundaries.")
        
        print(f"\nSystem box dimensions:")
        print(f"  Lx = {lx:.6f} Angstroms (X: {x_min:.6f} to {x_max:.6f})")
        print(f"  Ly = {ly:.6f} Angstroms (Y: {y_min:.6f} to {y_max:.6f})")
        print(f"  Lz = {lz:.6f} Angstroms (Z: {z_min:.6f} to {z_max_with_vacuum:.6f})")
        
        print(f"\nLattice parameters:")
        print(f"  a = {a:.6f} Angstroms")
        print(f"  b = {b:.6f} Angstroms")
        print(f"  c = {c:.6f} Angstroms")
        print(f"  alpha = {alpha:.2f} degrees")
        print(f"  beta = {beta:.2f} degrees")
        print(f"  gamma = {gamma:.2f} degrees")
        
        combined_comment = f"PTFE-water interface: {len(ptfe_atoms)} PTFE atoms, {num_water_atoms_final} water atoms, gap={args.gap:.2f} A"
        box_info = (lx, ly, lz, (x_min, x_max), (y_min, y_max), (z_min, z_max_with_vacuum), (a, b, c, alpha, beta, gamma))
        
        # Determine output file path
        if args.output_dir:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / args.output
        else:
            output_path = Path(args.output)
        
        # Write output
        print(f"\nWriting combined structure to: {output_path}")
        write_xyz(str(output_path), combined_atoms, combined_comment, box_info)
        
        print(f"  Total atoms: {len(combined_atoms)}")
        print(f"  PTFE atoms: {len(ptfe_atoms)}")
        print(f"  Water atoms: {num_water_atoms_final}")
        if args.gap < 0:
            print(f"  (Original water atoms: {len(water_atoms)}, removed: {len(water_atoms) - num_water_atoms_final})")
        print(f"\nDone!")
        print()
        print("=" * 70)
        print(f"Log file saved to: {log_path}")
    
    finally:
        # Restore original stdout and stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        tee_stdout.close()


if __name__ == '__main__':
    main()

