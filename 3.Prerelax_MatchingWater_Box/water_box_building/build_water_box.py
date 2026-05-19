#!/usr/bin/env python3
"""

This script reads the densified PDMS structure (LAMMPS data format) to extract
the simulation box dimensions in the X and Y directions, then constructs a water
box that:
  1. Matches the PDMS slab in X and Y (for seamless interface stacking).
  2. Has a Z dimension determined by EITHER:
     a) A user-specified number of water molecules (Lz is calculated to give
        the target density), OR
     b) A user-specified Lz value (the number of molecules is calculated).

The water molecules are placed on a simple cubic-like grid with small random
perturbations to break crystalline symmetry. The output is a standard XYZ file
that can be directly consumed by LAMMPS (read_data via conversion) or used as
input for further interface assembly.

Usage
-----
  python build_water_box.py                       # interactive prompts
  python build_water_box.py --n_water 100         # specify molecule count
  python build_water_box.py --lz 20.0             # specify Z height in Angstrom
  python build_water_box.py --lz 20.0 --density 1.0
  python build_water_box.py --pdms densified_pdms.lmpdat --output water_box.xyz

"""

import argparse
import math
import os
import random
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
AVOGADRO = 6.02214076e23           # mol^-1
WATER_MW = 18.01528                # g/mol  (H2O)
ANG3_PER_CM3 = 1e24               # Angstrom^3 per cm^3

# TIP4P/2005 geometry (used for placing H atoms relative to O)
OH_BOND_LENGTH = 0.9572            # Angstrom
HOH_ANGLE_DEG = 104.52             # degrees


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def parse_lmpdat_box(filepath: str) -> dict:
    """
    Parse a LAMMPS data file to extract the orthogonal box bounds.

    Returns a dict with keys: xlo, xhi, ylo, yhi, zlo, zhi, lx, ly, lz
    """
    box = {}
    pattern = re.compile(
        r"^\s*(-?\d+\.?\d*(?:[eE][+-]?\d+)?)\s+"
        r"(-?\d+\.?\d*(?:[eE][+-]?\d+)?)\s+"
        r"(xlo\s+xhi|ylo\s+yhi|zlo\s+zhi)",
        re.IGNORECASE,
    )
    with open(filepath, "r") as fh:
        for line in fh:
            m = pattern.match(line)
            if m:
                lo, hi = float(m.group(1)), float(m.group(2))
                label = m.group(3).strip().lower()
                if "xlo" in label:
                    box["xlo"], box["xhi"] = lo, hi
                elif "ylo" in label:
                    box["ylo"], box["yhi"] = lo, hi
                elif "zlo" in label:
                    box["zlo"], box["zhi"] = lo, hi

    for axis in ("x", "y", "z"):
        lo_key, hi_key = f"{axis}lo", f"{axis}hi"
        if lo_key not in box or hi_key not in box:
            raise ValueError(
                f"Could not find {lo_key}/{hi_key} in {filepath}. "
                "Is this a valid LAMMPS data file?"
            )
        box[f"l{axis}"] = box[hi_key] - box[lo_key]

    return box


def n_water_from_density(lx: float, ly: float, lz: float,
                         rho: float = 1.0) -> int:
    """
    Calculate the number of water molecules needed to fill a box
    (lx * ly * lz) in Angstrom at density rho (g/cm^3).
    """
    vol_ang3 = lx * ly * lz
    vol_cm3 = vol_ang3 / ANG3_PER_CM3
    mass_g = rho * vol_cm3
    n_mol = mass_g / WATER_MW
    n_molecules = n_mol * AVOGADRO
    return int(round(n_molecules))


def lz_from_n_water(lx: float, ly: float, n_water: int,
                    rho: float = 1.0) -> float:
    """
    Calculate the Z dimension needed for n_water molecules at density rho.
    """
    mass_g = (n_water / AVOGADRO) * WATER_MW
    vol_cm3 = mass_g / rho
    vol_ang3 = vol_cm3 * ANG3_PER_CM3
    lz = vol_ang3 / (lx * ly)
    return lz


def generate_water_molecule(ox: float, oy: float, oz: float) -> list:
    """
    Generate a single water molecule (O, H, H) centered at (ox, oy, oz)
    with a random orientation using the TIP4P/2005 geometry.

    Returns a list of 3 tuples: [("O", x, y, z), ("H", x, y, z), ("H", x, y, z)]
    """
    half_angle = math.radians(HOH_ANGLE_DEG / 2.0)

    # H positions in the molecular frame (O at origin, bisector along +z)
    h1_local = (
        0.0,
        OH_BOND_LENGTH * math.sin(half_angle),
        OH_BOND_LENGTH * math.cos(half_angle),
    )
    h2_local = (
        0.0,
        -OH_BOND_LENGTH * math.sin(half_angle),
        OH_BOND_LENGTH * math.cos(half_angle),
    )

    # Random rotation: generate Euler angles
    alpha = random.uniform(0, 2 * math.pi)
    beta = math.acos(random.uniform(-1, 1))
    gamma = random.uniform(0, 2 * math.pi)

    # Rotation matrix from Euler angles (ZYZ convention)
    ca, sa = math.cos(alpha), math.sin(alpha)
    cb, sb = math.cos(beta), math.sin(beta)
    cg, sg = math.cos(gamma), math.sin(gamma)

    R = [
        [ca * cb * cg - sa * sg, -ca * cb * sg - sa * cg, ca * sb],
        [sa * cb * cg + ca * sg, -sa * cb * sg + ca * cg, sa * sb],
        [-sb * cg,                sb * sg,                 cb],
    ]

    def rotate(v):
        return (
            R[0][0] * v[0] + R[0][1] * v[1] + R[0][2] * v[2],
            R[1][0] * v[0] + R[1][1] * v[1] + R[1][2] * v[2],
            R[2][0] * v[0] + R[2][1] * v[1] + R[2][2] * v[2],
        )

    h1_rot = rotate(h1_local)
    h2_rot = rotate(h2_local)

    atoms = [
        ("O", ox, oy, oz),
        ("H", ox + h1_rot[0], oy + h1_rot[1], oz + h1_rot[2]),
        ("H", ox + h2_rot[0], oy + h2_rot[1], oz + h2_rot[2]),
    ]
    return atoms


def build_water_box(lx: float, ly: float, lz: float,
                    n_water: int,
                    x_offset: float = 0.0,
                    y_offset: float = 0.0,
                    z_offset: float = 0.0,
                    margin: float = 1.0,
                    seed: int = 42) -> list:
    """
    Place n_water water molecules in a box of dimensions (lx, ly, lz)
    using a grid-based placement with random perturbations.

    Parameters
    ----------
    lx, ly, lz : float
        Box dimensions in Angstrom.
    n_water : int
        Number of water molecules to place.
    x_offset, y_offset, z_offset : float
        Origin offset for the box (lower-left corner).
    margin : float
        Minimum distance from box edges in Angstrom.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    list of tuples : [(element, x, y, z), ...]
    """
    random.seed(seed)

    # Effective box dimensions (with margin)
    eff_lx = lx - 2 * margin
    eff_ly = ly - 2 * margin
    eff_lz = lz - 2 * margin

    if eff_lx <= 0 or eff_ly <= 0 or eff_lz <= 0:
        raise ValueError(
            f"Box dimensions ({lx:.2f} x {ly:.2f} x {lz:.2f}) are too small "
            f"for margin={margin:.1f} A. Reduce margin or increase box size."
        )

    # Calculate grid dimensions
    # Target volume per molecule at the desired packing
    vol_per_mol = (eff_lx * eff_ly * eff_lz) / n_water
    spacing = vol_per_mol ** (1.0 / 3.0)

    nx = max(1, int(math.floor(eff_lx / spacing)))
    ny = max(1, int(math.floor(eff_ly / spacing)))
    nz = max(1, int(math.floor(eff_lz / spacing)))

    # Adjust grid to accommodate all molecules
    while nx * ny * nz < n_water:
        # Increase the dimension with the largest spacing
        dx = eff_lx / nx
        dy = eff_ly / ny
        dz = eff_lz / nz
        if dx >= dy and dx >= dz:
            nx += 1
        elif dy >= dz:
            ny += 1
        else:
            nz += 1

    # Actual spacing
    dx = eff_lx / nx
    dy = eff_ly / ny
    dz = eff_lz / nz

    # Generate grid positions and randomly select n_water of them
    grid_positions = []
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                cx = x_offset + margin + (ix + 0.5) * dx
                cy = y_offset + margin + (iy + 0.5) * dy
                cz = z_offset + margin + (iz + 0.5) * dz
                grid_positions.append((cx, cy, cz))

    random.shuffle(grid_positions)
    selected = grid_positions[:n_water]

    # Place water molecules with small random perturbation
    max_perturb = min(dx, dy, dz) * 0.15  # 15% of smallest spacing
    all_atoms = []
    for (cx, cy, cz) in selected:
        px = cx + random.uniform(-max_perturb, max_perturb)
        py = cy + random.uniform(-max_perturb, max_perturb)
        pz = cz + random.uniform(-max_perturb, max_perturb)
        mol_atoms = generate_water_molecule(px, py, pz)
        all_atoms.extend(mol_atoms)

    return all_atoms


def write_xyz(atoms: list, lx: float, ly: float, lz: float, filepath: str, comment: str = "") -> None:
    """Write atoms to an extended XYZ (extxyz) file format compatible with ASE."""
    with open(filepath, "w") as fh:
        fh.write(f"{len(atoms)}\n")
        
        # ASE extxyz format header
        lattice_str = f'Lattice="{lx:.6f} 0.0 0.0 0.0 {ly:.6f} 0.0 0.0 0.0 {lz:.6f}"'
        properties_str = "Properties=species:S:1:pos:R:3"
        extxyz_comment = f"{lattice_str} {properties_str}"
        if comment:
            extxyz_comment += f' info="{comment}"'
            
        fh.write(f"{extxyz_comment}\n")
        for (elem, x, y, z) in atoms:
            fh.write(f"{elem:2s} {x:16.8f} {y:16.8f} {z:16.8f}\n")


def write_lmpdat(atoms: list, n_water: int, box: dict,
                 filepath: str) -> None:
    """
    Write a minimal LAMMPS data file for the water box (atomic style).

    This produces an 'atomic' style data file with atom types:
      1 = O  (mass 15.9994)
      2 = H  (mass 1.0079)

    Note: For a full TIP4P simulation you would need bonds/angles/charges.
    This atomic-style output is suitable for use with pair_style mliap (MACE)
    which does not require topology information.
    """
    n_atoms = len(atoms)

    with open(filepath, "w") as fh:
        fh.write(f"LAMMPS data file — water box ({n_water} molecules), "
                 f"generated by build_water_box.py\n\n")
        fh.write(f"{n_atoms} atoms\n")
        fh.write("2 atom types\n\n")
        fh.write(f"{box['xlo']:.10f} {box['xhi']:.10f} xlo xhi\n")
        fh.write(f"{box['ylo']:.10f} {box['yhi']:.10f} ylo yhi\n")
        fh.write(f"{box['zlo']:.10f} {box['zhi']:.10f} zlo zhi\n\n")
        fh.write("Masses\n\n")
        fh.write("1 15.9994\n")
        fh.write("2  1.0079\n\n")
        fh.write("Atoms # atomic\n\n")

        type_map = {"O": 1, "H": 2}
        for i, (elem, x, y, z) in enumerate(atoms, start=1):
            atype = type_map[elem]
            fh.write(f"{i} {atype} {x:.10f} {y:.10f} {z:.10f}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build a water box matching PDMS slab XY dimensions. "
            "The water box is designed to be stacked on top of the PDMS slab "
            "along the Z axis for interfacial simulations."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python build_water_box.py --n_water 100\n"
            "  python build_water_box.py --lz 15.0\n"
            "  python build_water_box.py --lz 20.0 --density 0.997\n"
        ),
    )
    parser.add_argument(
        "--pdms", type=str,
        default="densified_pdms.lmpdat",
        help="Path to the PDMS LAMMPS data file (default: densified_pdms.lmpdat)",
    )
    parser.add_argument(
        "--n_water", type=int, default=100,
        help="Number of water molecules. If set, Lz is calculated from density.",
    )
    parser.add_argument(
        "--lz", type=float, default=None,
        help="Z dimension of the water box in Angstrom. "
             "If set, n_water is calculated from density.",
    )
    parser.add_argument(
        "--density", type=float, default=1.0,
        help="Target water density in g/cm^3 (default: 1.0)",
    )
    parser.add_argument(
        "--output", type=str, default="water_box",
        help="Output file basename without extension (default: water_box). "
             "Both .xyz and .lmpdat files are generated.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--margin", type=float, default=1.0,
        help="Minimum distance from box edges in Angstrom (default: 1.0)",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Read PDMS box dimensions
    # ------------------------------------------------------------------
    pdms_path = args.pdms
    if not os.path.isabs(pdms_path):
        pdms_path = os.path.join(os.path.dirname(__file__) or ".", pdms_path)

    print("=" * 60)
    print("  Water Box Builder for PDMS Interface Simulation")
    print("=" * 60)
    print(f"\n[1] Reading PDMS box from: {pdms_path}")

    pdms_box = parse_lmpdat_box(pdms_path)
    lx = pdms_box["lx"]
    ly = pdms_box["ly"]

    print(f"    PDMS box dimensions:")
    print(f"      Lx = {lx:.6f} A  (xlo={pdms_box['xlo']:.6f}, xhi={pdms_box['xhi']:.6f})")
    print(f"      Ly = {ly:.6f} A  (ylo={pdms_box['ylo']:.6f}, yhi={pdms_box['yhi']:.6f})")
    print(f"      Lz = {pdms_box['lz']:.6f} A  (zlo={pdms_box['zlo']:.6f}, zhi={pdms_box['zhi']:.6f})")
    print(f"    -> Water box will use Lx={lx:.4f}, Ly={ly:.4f} (matched)")

    # ------------------------------------------------------------------
    # 2. Determine Lz and n_water
    # ------------------------------------------------------------------
    rho = args.density

    if args.n_water is not None and args.lz is not None:
        # Both specified: use both, recalculate effective density
        n_water = args.n_water
        lz = args.lz
        actual_rho = (n_water / AVOGADRO * WATER_MW) / (lx * ly * lz / ANG3_PER_CM3)
        print(f"\n[2] Both --n_water and --lz specified:")
        print(f"    n_water = {n_water}")
        print(f"    Lz      = {lz:.4f} A")
        print(f"    Effective density = {actual_rho:.4f} g/cm^3")
    elif args.n_water is not None:
        n_water = args.n_water
        lz = lz_from_n_water(lx, ly, n_water, rho)
        print(f"\n[2] Computing Lz from n_water={n_water} at density={rho:.4f} g/cm^3:")
        print(f"    Lz = {lz:.6f} A")
    elif args.lz is not None:
        lz = args.lz
        n_water = n_water_from_density(lx, ly, lz, rho)
        print(f"\n[2] Computing n_water from Lz={lz:.4f} A at density={rho:.4f} g/cm^3:")
        print(f"    n_water = {n_water}")
    else:
        # Interactive mode: ask user
        print(f"\n[2] No --n_water or --lz specified. Enter one of:")
        print(f"    (a) Number of water molecules, or")
        print(f"    (b) Z dimension in Angstrom")
        choice = input("    Enter 'a' or 'b': ").strip().lower()
        if choice == "a":
            n_water = int(input("    Number of water molecules: "))
            lz = lz_from_n_water(lx, ly, n_water, rho)
            print(f"    -> Lz = {lz:.6f} A (at density {rho:.4f} g/cm^3)")
        elif choice == "b":
            lz = float(input("    Z dimension (Angstrom): "))
            n_water = n_water_from_density(lx, ly, lz, rho)
            print(f"    -> n_water = {n_water} (at density {rho:.4f} g/cm^3)")
        else:
            print("    Invalid choice. Exiting.")
            sys.exit(1)

    # Validate
    if n_water <= 0:
        print(f"ERROR: n_water={n_water} is not positive. Check your inputs.")
        sys.exit(1)
    if lz <= 0:
        print(f"ERROR: Lz={lz:.4f} is not positive. Check your inputs.")
        sys.exit(1)

    n_atoms = n_water * 3
    vol_ang3 = lx * ly * lz
    actual_density = (n_water / AVOGADRO * WATER_MW) / (vol_ang3 / ANG3_PER_CM3)

    print(f"\n    --- Water Box Summary ---")
    print(f"    Lx x Ly x Lz  = {lx:.4f} x {ly:.4f} x {lz:.4f} A")
    print(f"    Volume         = {vol_ang3:.2f} A^3")
    print(f"    N(H2O)         = {n_water}")
    print(f"    N(atoms)       = {n_atoms}")
    print(f"    Target density = {rho:.4f} g/cm^3")
    print(f"    Actual density = {actual_density:.4f} g/cm^3")

    # ------------------------------------------------------------------
    # 3. Build water box
    # ------------------------------------------------------------------
    print(f"\n[3] Placing {n_water} water molecules on grid ...")

    # Use the same xlo, ylo as PDMS, and place water box starting from
    # pdms_zhi (on top of PDMS)
    x_off = pdms_box["xlo"]
    y_off = pdms_box["ylo"]
    z_off = 0.0  # Water box starts at z=0 (will be shifted during assembly)

    all_atoms = build_water_box(
        lx=lx, ly=ly, lz=lz,
        n_water=n_water,
        x_offset=x_off,
        y_offset=y_off,
        z_offset=z_off,
        margin=args.margin,
        seed=args.seed,
    )

    print(f"    Placed {len(all_atoms)} atoms ({n_water} H2O molecules)")

    # ------------------------------------------------------------------
    # 4. Write output files
    # ------------------------------------------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    basename = args.output

    xyz_path = os.path.join(script_dir, f"{basename}.xyz")
    lmpdat_path = os.path.join(script_dir, f"{basename}.lmpdat")

    # XYZ file (ASE extxyz format)
    comment = (
        f"Water box: {n_water} H2O, "
        f"density={actual_density:.4f} g/cm^3"
    )
    write_xyz(all_atoms, lx, ly, lz, xyz_path, comment=comment)
    print(f"\n[4] Output files written:")
    print(f"    XYZ:    {xyz_path}")

    # LAMMPS data file
    water_box = {
        "xlo": x_off, "xhi": x_off + lx,
        "ylo": y_off, "yhi": y_off + ly,
        "zlo": z_off, "zhi": z_off + lz,
    }
    write_lmpdat(all_atoms, n_water, water_box, lmpdat_path)
    print(f"    LMPDAT: {lmpdat_path}")

    # ------------------------------------------------------------------
    # 5. Print assembly instructions
    # ------------------------------------------------------------------
    pdms_zhi = pdms_box["zhi"]
    water_zlo_shifted = pdms_zhi
    water_zhi_shifted = pdms_zhi + lz

    print(f"\n{'=' * 60}")
    print(f"  Assembly Instructions")
    print(f"{'=' * 60}")
    print(f"  To stack the water box on top of the PDMS slab:")
    print(f"    1. PDMS slab occupies z = [{pdms_box['zlo']:.4f}, {pdms_zhi:.4f}] A")
    print(f"    2. Shift water box z-coordinates by +{pdms_zhi:.4f} A")
    print(f"       -> Water box will occupy z = [{water_zlo_shifted:.4f}, {water_zhi_shifted:.4f}] A")
    print(f"    3. Total system height = {water_zhi_shifted - pdms_box['zlo']:.4f} A")
    print(f"    4. Consider adding a vacuum gap (~10-15 A) above the water")
    print(f"       for slab-model simulations with surface boundary conditions.")
    print(f"{'=' * 60}")
    print(f"  Done!")


if __name__ == "__main__":
    main()
