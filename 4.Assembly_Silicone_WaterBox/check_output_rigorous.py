import numpy as np
from ase.io import read

def run_rigorous_check():
    xyz_path = "/Users/siqi/GitHub/H2O2_interfacial_water/4.Assembly_Silicone_WaterBox/charged_interface/assembled_interface.xyz"
    print("Reading assembled system...")
    atoms = read(xyz_path)
    
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    cell = atoms.get_cell()
    Lx, Ly, Lz = cell.lengths()
    print(f"Total atoms: {len(atoms)}")
    print(f"Cell dimensions: Lx={Lx:.4f}, Ly={Ly:.4f}, Lz={Lz:.4f}")
    
    n_pdms = 523
    pdms_positions = positions[:n_pdms]
    water_positions = positions[n_pdms:]
    water_symbols = symbols[n_pdms:]
    
    print("\n--- Checking PDMS coordinates ---")
    min_xyz = np.min(pdms_positions, axis=0)
    max_xyz = np.max(pdms_positions, axis=0)
    print(f"PDMS X range: [{min_xyz[0]:.4f}, {max_xyz[0]:.4f}]")
    print(f"PDMS Y range: [{min_xyz[1]:.4f}, {max_xyz[1]:.4f}]")
    print(f"PDMS Z range: [{min_xyz[2]:.4f}, {max_xyz[2]:.4f}]")
    
    print("\n--- Checking Water molecules ---")
    # Identify water molecules and their ionization state
    # We group by 3 because water molecules were appended as O, H, H sequentially
    num_water_mols = len(water_positions) // 3
    print(f"Total water molecules: {num_water_mols}")
    
    oh_indices = []
    h3o_indices = []
    neutral_indices = []
    
    broken_neutral_count = 0
    
    for i in range(num_water_mols):
        o_idx = n_pdms + i * 3
        h1_idx = o_idx + 1
        h2_idx = o_idx + 2
        
        # Verify symbols
        assert symbols[o_idx] == 'O', f"Atom at {o_idx} is {symbols[o_idx]}, not O"
        assert symbols[h1_idx] == 'H', f"Atom at {h1_idx} is {symbols[h1_idx]}, not H"
        assert symbols[h2_idx] == 'H', f"Atom at {h2_idx} is {symbols[h2_idx]}, not H"
        
        # Let's count how many hydrogens are physically bonded to this Oxygen
        # Standard O-H bond is ~0.96 A, so let's check all H positions in the system
        o_pos = positions[o_idx]
        
        # Calculate distances without X-Y PBC but WITH intrinsic Z PBC of the original water box
        # because the raw xyz might contain Z-wrapped molecules.
        water_Lz = 8.4436268476
        diff1 = positions[h1_idx] - o_pos
        diff1[2] = diff1[2] - np.round(diff1[2] / water_Lz) * water_Lz
        d1_no_pbc = np.linalg.norm(diff1)
        
        diff2 = positions[h2_idx] - o_pos
        diff2[2] = diff2[2] - np.round(diff2[2] / water_Lz) * water_Lz
        d2_no_pbc = np.linalg.norm(diff2)
        
        # Let's check distance to all H atoms in the system with X-Y PBC and intrinsic Z PBC
        h_dists = []
        for h_idx in range(n_pdms, len(atoms)):
            if symbols[h_idx] != 'H': continue
            diff = positions[h_idx] - o_pos
            # Apply X-Y PBC of the combined supercell
            diff[0] = diff[0] - np.round(diff[0] / Lx) * Lx
            diff[1] = diff[1] - np.round(diff[1] / Ly) * Ly
            # Apply intrinsic Z PBC of the water slab
            diff[2] = diff[2] - np.round(diff[2] / water_Lz) * water_Lz
            dist = np.linalg.norm(diff)
            if dist < 1.3:
                h_dists.append((h_idx, dist))
                
        # Classify by number of bonded H atoms
        bonded_count = len(h_dists)
        if bonded_count == 1:
            oh_indices.append((o_idx, h_dists))
        elif bonded_count == 2:
            neutral_indices.append((o_idx, h_dists, d1_no_pbc, d2_no_pbc))
        elif bonded_count == 3:
            h3o_indices.append((o_idx, h_dists))
        else:
            print(f"WARNING: Water molecule {i} has unexpected bonded H count: {bonded_count}!")
            
    print(f"Found {len(neutral_indices)} neutral H2O molecules.")
    print(f"Found {len(oh_indices)} OH- ions.")
    print(f"Found {len(h3o_indices)} H3O+ ions.")
    
    # Verify exact counts
    assert len(oh_indices) == 2, f"Expected 2 OH- ions, found {len(oh_indices)}"
    assert len(h3o_indices) == 2, f"Expected 2 H3O+ ions, found {len(h3o_indices)}"
    print("\nSUCCESS: Species count exactly matches the expected 2 EDL pairs!\n")
    
    # Check bond lengths for neutral water molecules
    broken_neutral = []
    for o_idx, h_dists, d1, d2 in neutral_indices:
        if d1 > 1.1 or d2 > 1.1:
            broken_neutral.append((o_idx, d1, d2))
            
    print(f"Neutral H2O molecules with O-H bond > 1.1 A (without PBC): {len(broken_neutral)}")
    for o_idx, d1, d2 in broken_neutral:
        print(f"  O idx: {o_idx}, O-H1: {d1:.4f} A, O-H2: {d2:.4f} A")
        
    # Check OH- details
    print("\n--- OH- Details ---")
    for o_idx, h_dists in oh_indices:
        h_idx, dist = h_dists[0]
        # Let's compute distance to PDMS slab (with X-Y PBC)
        o_pos = positions[o_idx]
        diffs = pdms_positions - o_pos
        diffs[:, 0] = diffs[:, 0] - np.round(diffs[:, 0] / Lx) * Lx
        diffs[:, 1] = diffs[:, 1] - np.round(diffs[:, 1] / Ly) * Ly
        pdms_dists = np.linalg.norm(diffs, axis=1)
        min_pdms_dist = np.min(pdms_dists)
        
        print(f"  OH- Oxygen idx {o_idx} at Z={o_pos[2]:.4f} A:")
        print(f"    Bonded H idx: {h_idx}, Bond length: {dist:.4f} A")
        print(f"    Min 3D distance to PDMS atoms: {min_pdms_dist:.4f} A")
        
    # Check H3O+ details
    print("\n--- H3O+ Details ---")
    for o_idx, h_dists in h3o_indices:
        o_pos = positions[o_idx]
        h_info = ", ".join([f"H_idx {h[0]} ({h[1]:.4f} A)" for h in h_dists])
        print(f"  H3O+ Oxygen idx {o_idx} at Z={o_pos[2]:.4f} A:")
        print(f"    Bonded H list: {h_info}")
        
    # Let's check the minimum distance of ALL water oxygen atoms to PDMS
    # to see if the OH- are indeed the two closest ones.
    print("\n--- Verifying interface selection (lowest min_dist_to_pdms) ---")
    all_o_dists = []
    for i in range(num_water_mols):
        o_idx = n_pdms + i * 3
        o_pos = positions[o_idx]
        diffs = pdms_positions - o_pos
        diffs[:, 0] = diffs[:, 0] - np.round(diffs[:, 0] / Lx) * Lx
        diffs[:, 1] = diffs[:, 1] - np.round(diffs[:, 1] / Ly) * Ly
        min_pdms_dist = np.min(np.linalg.norm(diffs, axis=1))
        
        # Check if it's OH- or H3O+ or neutral
        is_oh = any(o_idx == item[0] for item in oh_indices)
        is_h3o = any(o_idx == item[0] for item in h3o_indices)
        all_o_dists.append((o_idx, min_pdms_dist, 'OH-' if is_oh else ('H3O+' if is_h3o else 'H2O')))
        
    # Sort by minimum distance to PDMS
    all_o_dists.sort(key=lambda x: x[1])
    print("Closest 10 water molecules to PDMS slab (by 3D minimum distance):")
    for idx, dist, label in all_o_dists[:10]:
        print(f"  O idx: {idx}, Distance: {dist:.4f} A, Type: {label}")
        
    # Check if OH- are indeed the closest ones
    closest_types = [item[2] for item in all_o_dists[:2]]
    print(f"Are the 2 closest water molecules converted to OH-? {closest_types == ['OH-', 'OH-']}")

if __name__ == '__main__':
    run_rigorous_check()
