#!/usr/bin/env python
"""
Build a simplified PDMS-like slab with optional peroxide-cure crosslinking.

Crosslinking chemistry (peroxide cure):
  - Curing agent: 2,5-dimethyl-2,5-di(tert-butylperoxy)hexane (1%)
  - Mechanism: radical abstraction of H from methyl groups, then C-C coupling
  - Result: Si-CH2-CH2-Si bridges between different chains

Usage examples:
  # Linear PDMS (no crosslink)
  python silicone_slab.py --chain-length 6 --rows 3 --layers 3

  # With peroxide-cure crosslinking (5 crosslinks)
  python silicone_slab.py --crosslink-mode peroxide --crosslink-count 5 --seed 42

  # With crosslink density target (3% of methyls participate)
  python silicone_slab.py --crosslink-mode peroxide --crosslink-density 0.03  --seed 42 --output silicone_slab_to_melting.xyz
"""

import os
import argparse
import numpy as np
from ase import Atom, Atoms
from ase.io import write


# =============================================================================
# Chain builder with methyl tracking
# =============================================================================

def build_silicone_chain(length: int = 5):
    """
    Build a single simplified PDMS chain: -[Si(CH3)2-O]-n

    Returns:
        atoms: ASE Atoms object
        methyl_map: dict mapping si_local_idx -> list of (c_idx, [h1, h2, h3])
    """
    atoms = Atoms()
    methyl_map = {}

    si_o_bond = 1.63
    z_shift = si_o_bond * 2
    si_c_dist = 1.87
    c_h_dist = 1.09
    tetra_angle = np.arccos(-1.0 / 3.0)

    for i in range(length):
        z_si = i * z_shift
        si_idx = len(atoms)
        atoms.append(Atom("Si", position=(0.0, 0.0, z_si)))

        if i < length - 1:
            z_o = z_si + si_o_bond
            atoms.append(Atom("O", position=(0.0, 0.0, z_o)))

        theta = i * np.pi / 4
        methyl_map[si_idx] = []

        def _add_methyl(c_phi):
            c_x = si_c_dist * np.sin(tetra_angle) * np.cos(c_phi)
            c_y = si_c_dist * np.sin(tetra_angle) * np.sin(c_phi)
            c_z = z_si + si_c_dist * np.cos(tetra_angle)

            c_idx = len(atoms)
            atoms.append(Atom("C", position=(c_x, c_y, c_z)))

            h_dir = np.array([c_x, c_y, c_z]) - np.array([0.0, 0.0, z_si])
            h_dir = h_dir / np.linalg.norm(h_dir)

            perp1 = np.array([-h_dir[1], h_dir[0], 0.0])
            if np.linalg.norm(perp1) < 0.1:
                perp1 = np.array([0.0, -h_dir[2], h_dir[1]])
            perp1 = perp1 / np.linalg.norm(perp1)
            perp2 = np.cross(h_dir, perp1)
            perp2 = perp2 / np.linalg.norm(perp2)

            h_indices = []
            h_pos_base = np.array([c_x, c_y, c_z])
            for h_idx in range(3):
                angle_h = h_idx * 2 * np.pi / 3
                h_offset = c_h_dist * (np.cos(angle_h) * perp1 + np.sin(angle_h) * perp2)
                h_indices.append(len(atoms))
                atoms.append(Atom("H", position=h_pos_base + h_offset))

            methyl_map[si_idx].append((c_idx, h_indices))

        _add_methyl(theta)
        _add_methyl(theta + np.pi)

    return atoms, methyl_map


# =============================================================================
# Crosslinking functions
# =============================================================================

def _find_methyl_carbons(atoms: Atoms, methyl_map_global: dict):
    """
    Return list of (c_idx, si_idx, chain_tag, h_indices) for all tracked methyls.
    """
    result = []
    tags = atoms.get_tags()
    for si_idx, groups in methyl_map_global.items():
        chain_tag = tags[si_idx]
        for c_idx, h_indices in groups:
            result.append((c_idx, si_idx, chain_tag, h_indices))
    return result


def add_peroxide_crosslinks(
    slab: Atoms,
    methyl_map_global: dict,
    crosslink_count: int = 0,
    crosslink_density: float = 0.0,
    d_min: float = 3.5,
    d_max: float = 5.5,
    cc_bond: float = 1.54,
    seed: int = 0,
):
    """
    Add peroxide-cure style C-C crosslinks between methyls on different chains.

    For each crosslink:
      - Select two methyl carbons from different chains within [d_min, d_max]
      - Remove 1 H from each methyl (simulating H abstraction by radicals)
      - Move the two C atoms to form a C-C bond (~1.54 A)

    Args:
        slab: ASE Atoms (will be modified in place, then rebuilt)
        methyl_map_global: dict[si_global_idx] -> list of (c_idx, [h1, h2, h3])
        crosslink_count: number of crosslinks to add (if > 0, overrides density)
        crosslink_density: fraction of methyls to involve in crosslinking (0-1)
        d_min, d_max: distance window for initial C...C selection
        cc_bond: target C-C bond length
        seed: random seed

    Returns:
        new_slab: rebuilt Atoms with crosslinks added
        stats: dict with crosslink statistics
    """
    rng = np.random.default_rng(seed)

    # Gather all methyl info
    methyls = _find_methyl_carbons(slab, methyl_map_global)
    total_methyls = len(methyls)

    if total_methyls < 2:
        print("Warning: Not enough methyls for crosslinking.")
        return slab, {"requested": 0, "added": 0}

    # Determine target crosslink count
    if crosslink_count > 0:
        target = crosslink_count
    elif crosslink_density > 0:
        # Each crosslink uses 2 methyls
        target = max(1, int(total_methyls * crosslink_density / 2))
    else:
        return slab, {"requested": 0, "added": 0}

    pos = slab.get_positions().copy()

    # Build candidate pairs: different chains, distance in window
    candidates = []
    for i in range(len(methyls)):
        c_i, si_i, tag_i, h_i = methyls[i]
        for j in range(i + 1, len(methyls)):
            c_j, si_j, tag_j, h_j = methyls[j]
            if tag_i == tag_j:
                continue  # same chain, skip
            d = np.linalg.norm(pos[c_j] - pos[c_i])
            if d_min <= d <= d_max:
                candidates.append((d, i, j))

    if not candidates:
        print(f"Warning: No methyl pairs found in distance window [{d_min}, {d_max}] A.")
        print("Try increasing d_max or adjusting chain packing.")
        return slab, {"requested": target, "added": 0}

    # Sort by distance (prefer closer pairs for more realistic crosslinks)
    candidates.sort(key=lambda x: x[0])
    # Add some randomness while keeping preference for closer pairs
    top_k = min(len(candidates), max(20, 5 * target))
    pick_pool = candidates[:top_k]
    rng.shuffle(pick_pool)

    # Track which methyls are used
    used_methyl_idx = set()
    # Track atoms to delete (H atoms)
    atoms_to_delete = set()
    # Track position adjustments for C atoms
    c_adjustments = {}

    added = 0
    for d, i, j in pick_pool:
        if added >= target:
            break
        if i in used_methyl_idx or j in used_methyl_idx:
            continue

        c_i, si_i, tag_i, h_list_i = methyls[i]
        c_j, si_j, tag_j, h_list_j = methyls[j]

        # Mark one H from each methyl for deletion
        atoms_to_delete.add(h_list_i[0])
        atoms_to_delete.add(h_list_j[0])

        # Calculate new positions: move both C atoms toward midpoint
        r_ci = pos[c_i].copy()
        r_cj = pos[c_j].copy()
        midpoint = (r_ci + r_cj) / 2.0
        direction = r_cj - r_ci
        dist = np.linalg.norm(direction)
        if dist < 1e-6:
            continue
        direction = direction / dist

        # New positions: each C moves toward midpoint to achieve cc_bond
        new_ci = midpoint - direction * (cc_bond / 2.0)
        new_cj = midpoint + direction * (cc_bond / 2.0)

        c_adjustments[c_i] = new_ci
        c_adjustments[c_j] = new_cj

        used_methyl_idx.add(i)
        used_methyl_idx.add(j)
        added += 1

    # Now rebuild slab:
    # 1. Apply C position adjustments
    # 2. Remove deleted H atoms (and update remaining H positions if needed)

    new_pos = pos.copy()
    for c_idx, new_c_pos in c_adjustments.items():
        old_c_pos = pos[c_idx]
        delta = new_c_pos - old_c_pos
        new_pos[c_idx] = new_c_pos

        # Also move the remaining H atoms attached to this C
        # Find which methyl this C belongs to
        for si_idx, groups in methyl_map_global.items():
            for c, h_list in groups:
                if c == c_idx:
                    for h_idx in h_list:
                        if h_idx not in atoms_to_delete:
                            new_pos[h_idx] = pos[h_idx] + delta
                    break

    # Build keep mask
    keep = [i not in atoms_to_delete for i in range(len(slab))]
    keep_indices = [i for i in range(len(slab)) if keep[i]]

    # Create new atoms
    new_slab = Atoms()
    new_slab.set_cell(slab.get_cell())
    new_slab.pbc = slab.pbc

    for new_idx, old_idx in enumerate(keep_indices):
        atom = slab[old_idx]
        atom.position = new_pos[old_idx]
        new_slab.append(atom)

    stats = {
        "requested": target,
        "added": added,
        "methyls_total": total_methyls,
        "methyls_crosslinked": 2 * added,
        "crosslink_fraction": (2 * added) / total_methyls if total_methyls > 0 else 0,
        "h_atoms_removed": len(atoms_to_delete),
    }

    return new_slab, stats


def compute_density(atoms: Atoms) -> float:
    """
    Compute mass density in g/cm^3.
    """
    from ase.data import atomic_masses, atomic_numbers

    total_mass_amu = 0.0
    for atom in atoms:
        z = atomic_numbers[atom.symbol]
        total_mass_amu += atomic_masses[z]

    # Volume in Angstrom^3
    cell = atoms.get_cell()
    volume_A3 = abs(np.linalg.det(cell))

    if volume_A3 < 1e-6:
        return 0.0

    # Convert: 1 amu = 1.66054e-24 g, 1 A^3 = 1e-24 cm^3
    # density = (mass_amu * 1.66054e-24 g) / (volume_A3 * 1e-24 cm^3)
    #         = mass_amu * 1.66054 / volume_A3  [g/cm^3]
    density = total_mass_amu * 1.66054 / volume_A3

    return density


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Build PDMS slab with optional peroxide-cure crosslinking.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Chain/slab parameters
    parser.add_argument("--chain-length", type=int, default=6,
                        help="Number of Si-O units per chain (default: 6)")
    parser.add_argument("--rows", type=int, default=3,
                        help="Number of chains per layer in x direction (default: 3)")
    parser.add_argument("--layers", type=int, default=3,
                        help="Number of layers in z direction (default: 3)")
    parser.add_argument("--a-dist", type=float, default=5.5,
                        help="Inter-chain spacing in Angstrom (default: 5.5)")
    parser.add_argument("--cell-z", type=float, default=20.0,
                        help="Cell height in z direction (default: 20.0)")

    # Crosslink parameters
    parser.add_argument("--crosslink-mode", choices=["none", "peroxide"], default="none",
                        help="Crosslink mode: none (linear) or peroxide (C-C bridges)")
    parser.add_argument("--crosslink-count", type=int, default=0,
                        help="Number of crosslinks to add (overrides --crosslink-density)")
    parser.add_argument("--crosslink-density", type=float, default=0.0,
                        help="Fraction of methyls to crosslink (0-1, e.g. 0.03 for 3%%)")
    parser.add_argument("--crosslink-dmin", type=float, default=3.5,
                        help="Min C...C distance for crosslink candidate (default: 3.5)")
    parser.add_argument("--crosslink-dmax", type=float, default=5.5,
                        help="Max C...C distance for crosslink candidate (default: 5.5)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for crosslink selection")

    # Output
    parser.add_argument("--output", type=str, default="",
                        help="Output XYZ filename (default: silicone_slab.xyz in script dir)")

    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # 1. Build single chain with methyl tracking
    # -------------------------------------------------------------------------
    chain, chain_methyl_map = build_silicone_chain(length=args.chain_length)
    chain.center(vacuum=2.0)

    # -------------------------------------------------------------------------
    # 2. Pack chains into slab with chain tagging
    # -------------------------------------------------------------------------
    slab = Atoms()
    methyl_map_global = {}
    chain_id = 0

    a_dist = args.a_dist
    cell_x = 4 * a_dist
    chain_span = np.ptp(chain.get_positions()[:, 2]) + 1.0
    cell_y = chain_span
    cell_z = args.cell_z

    for layer in range(args.layers):
        for row in range(args.rows):
            new_chain = chain.copy()

            # Rotate chain: original grows along z, we want it along y
            new_chain.rotate(90, "x", rotate_cell=False)

            # Calculate offset
            x_offset = row * a_dist
            if layer % 2 == 1:
                x_offset += a_dist / 2.0  # Hexagonal packing offset

            z_offset = layer * (a_dist * np.sin(np.pi / 3.0))

            new_chain.translate([x_offset, 0.0, z_offset])

            # Tag all atoms in this chain
            tags = new_chain.get_tags()
            tags[:] = chain_id
            new_chain.set_tags(tags)

            # Merge methyl_map with index offset
            offset = len(slab)
            for si_local, groups in chain_methyl_map.items():
                si_global = si_local + offset
                methyl_map_global[si_global] = [
                    (c + offset, [h + offset for h in hs]) for (c, hs) in groups
                ]

            slab.extend(new_chain)
            chain_id += 1

    # Set cell and center
    slab.set_cell([cell_x, cell_y, cell_z])
    slab.center()

    # Move slab to bottom
    z_min = float(np.min(slab.positions[:, 2]))
    slab.translate([0.0, 0.0, -z_min + 1.0])

    slab.pbc = [True, True, True]

    total_chains = args.rows * args.layers
    total_si = args.chain_length * total_chains
    total_methyls = 2 * total_si

    print("=" * 60)
    print("PDMS Slab Construction")
    print("=" * 60)
    print(f"Chain length:      {args.chain_length} Si-O units")
    print(f"Rows x Layers:     {args.rows} x {args.layers} = {total_chains} chains")
    print(f"Total Si atoms:    {total_si}")
    print(f"Total methyls:     {total_methyls}")
    print(f"Atoms before xlink:{len(slab)}")

    # -------------------------------------------------------------------------
    # 3. Apply crosslinking if requested
    # -------------------------------------------------------------------------
    crosslink_stats = {"added": 0}

    if args.crosslink_mode == "peroxide":
        count = args.crosslink_count
        density = args.crosslink_density

        if count <= 0 and density <= 0:
            print("\nWarning: --crosslink-mode peroxide but no count/density specified.")
            print("Use --crosslink-count N or --crosslink-density 0.03 (3%)")
        else:
            print(f"\nApplying peroxide-cure crosslinking...")
            print(f"  Distance window: [{args.crosslink_dmin}, {args.crosslink_dmax}] A")

            slab, crosslink_stats = add_peroxide_crosslinks(
                slab,
                methyl_map_global,
                crosslink_count=count,
                crosslink_density=density,
                d_min=args.crosslink_dmin,
                d_max=args.crosslink_dmax,
                seed=args.seed,
            )

            print(f"  Crosslinks requested: {crosslink_stats['requested']}")
            print(f"  Crosslinks added:     {crosslink_stats['added']}")
            print(f"  Methyls crosslinked:  {crosslink_stats['methyls_crosslinked']} / {crosslink_stats['methyls_total']}")
            print(f"  Crosslink fraction:   {crosslink_stats['crosslink_fraction']:.1%}")
            print(f"  H atoms removed:      {crosslink_stats['h_atoms_removed']}")

    # -------------------------------------------------------------------------
    # 4. Compute and report final properties
    # -------------------------------------------------------------------------
    density = compute_density(slab)

    print(f"\nFinal structure:")
    print(f"  Atoms:   {len(slab)}")
    print(f"  Cell:    {slab.get_cell()[0,0]:.2f} x {slab.get_cell()[1,1]:.2f} x {slab.get_cell()[2,2]:.2f} A")
    print(f"  Density: {density:.3f} g/cm^3")

    if args.crosslink_mode == "peroxide" and crosslink_stats["added"] > 0:
        print(f"\n  NOTE: This is a geometry-only crosslinked structure.")
        print(f"        Relax with MACE at curing temperature (443 K) before use.")

    # -------------------------------------------------------------------------
    # 5. Save output
    # -------------------------------------------------------------------------
    if args.output:
        output_file = args.output
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_file = os.path.join(script_dir, "silicone_slab.xyz")

    write(output_file, slab)
    print(f"\nOutput file: {output_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
