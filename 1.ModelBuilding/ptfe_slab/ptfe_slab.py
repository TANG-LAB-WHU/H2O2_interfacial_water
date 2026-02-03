#! /usr/bin/env python

import os
import numpy as np
from ase import Atom, Atoms
from ase.build import make_supercell
from ase.visualize import view
from ase.io import write

def build_ptfe_chain(length=5):
    """
    Build a single simplified helical PTFE chain (-CF2-)n.
    
    Parameters:
    length: Number of CF2 units
    
    Note: This is a simplified helical structure.
    """
    atoms = Atoms()
    # C-C bond length ~1.54A, C-F ~1.35A, F-C-F ~108 degrees
    # Grow along the z-axis
    z_shift = 1.3  # Increment along z-axis for each CF2 unit
    
    for i in range(length):
        theta = i * (np.pi * 14 / 15) # Simulate helical twist (15/7 helix approximation)
        z = i * z_shift
        
        # Carbon
        atoms.append(Atom('C', position=(0, 0, z)))
        
        # Fluorine 1
        fx1 = 1.35 * np.cos(theta)
        fy1 = 1.35 * np.sin(theta)
        atoms.append(Atom('F', position=(fx1, fy1, z)))
        
        # Fluorine 2
        fx2 = 1.35 * np.cos(theta + np.pi * 1.3) # Handle F-C-F angle
        fy2 = 1.35 * np.sin(theta + np.pi * 1.3)
        atoms.append(Atom('F', position=(fx2, fy2, z)))
        
    return atoms

# 1. Generate a single long chain
# Assume we want to build a box approx 30A long; with z_shift 1.3 per CF2, we need about 24 units
chain = build_ptfe_chain(length=12)
chain.center(vacuum=2.0) # Temporarily center

# 2. Define unit cell (Hexagonal Packing)
# Chain spacing approx 5.66 A
a_dist = 5.66
# Simple hexagonal packing logic:
# Manually replicate chains to build the slab

slab = Atoms()
cell_x = 6 * a_dist  # Replicate 6 times in x direction ~ 34 A
cell_y = chain.get_positions()[-1, 2] + 1.0 # Length in y direction (original chain length)
cell_z = 20.0 # Initial thickness

# Build two layers of chains (Layer 1 and Layer 2)
rows = 3   # Number of chains in x direction
layers = 3 # Number of layers in z direction

for layer in range(layers):
    for row in range(rows):
        new_chain = chain.copy()
        
        # Calculate offset
        x_offset = row * a_dist
        if layer % 2 == 1:
            x_offset += a_dist / 2.0 # Hexagonal close packing offset
            
        z_offset = layer * (a_dist * np.sin(np.pi/3)) # Interlayer spacing
        
        # Rotate chain to lie in xy plane (original chain grows along z)
        # Here we need the chain to extend along the y-axis
        new_chain.rotate('z', 'y', rotate_cell=False) 
        
        # Translate position
        new_chain.translate([x_offset, 0, z_offset])
        
        slab.extend(new_chain)

# 3. Set unit cell and center
slab.set_cell([cell_x, cell_y, cell_z])
slab.center(axis=2) # Center in Z direction, or place at the bottom

# Move Slab to the bottom, leaving vacuum above for the droplet
z_min = np.min(slab.positions[:, 2])
slab.translate([0, 0, -z_min + 1.0]) 

# 4. Periodic boundary conditions
slab.pbc = [True, True, True]

# 5. Save
# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
output_file = os.path.join(script_dir, 'ptfe_slab.xyz')
write(output_file, slab)
print(f"PTFE Slab built. Size: {slab.get_cell()}")
print(f"Number of atoms: {len(slab)}")
print(f"Output file: {output_file}")