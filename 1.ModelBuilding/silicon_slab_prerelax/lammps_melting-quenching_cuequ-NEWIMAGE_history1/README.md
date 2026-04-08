# PDMS Melt-Quench Densification Simulation

This directory contains scripts for running LAMMPS molecular dynamics simulations to densify crosslinked PDMS (polydimethylsiloxane) rubber using melt-quench protocol.

## Purpose

Transform a geometry-only crosslinked PDMS structure (density ~0.76 g/cm³) into a realistic densified structure (target density ~1.10 g/cm³) by simulating the curing process.

## Chemistry Context

- **Material**: Crosslinked PDMS (silicone rubber)
- **Curing agent**: 2,5-dimethyl-2,5-di(tert-butylperoxy)hexane (1%)
- **Curing conditions**: 170°C (443 K), 10 min
- **Target density**: 1.10 g/cm³

## Simulation Workflow

The melt-quench protocol mimics the curing process:

1. **Energy Minimization**: Remove bad contacts from initial geometry
2. **Heating (300K → 443K)**: Heat to curing temperature
3. **NPT Equilibration at 443K**: Allow chain relaxation at high temperature
4. **Cooling (443K → 300K)**: Slow cooling to room temperature
5. **Final NPT at 300K**: Equilibrate at target temperature
6. **Output**: Densified structure for interface simulations

## Files

| File | Description |
|------|-------------|
| `silicone_slab.xyz` | Input structure (crosslinked PDMS from silicone_slab.py) |
| `xyz_to_lammps.py` | Convert ASE-style XYZ to LAMMPS data format |
| `convert_mace_to_mliap.py` | Convert MACE model to MLIAP format |
| `lammps_melt_quench.inp` | LAMMPS input script for melt-quench |
| `docker-compose.yml` | Docker configuration for LAMMPS with GPU |
| `run_simulation.bat` | Windows batch script to run full workflow |

## Quick Start

### Prerequisites

- Docker Desktop with NVIDIA GPU support
- MACE pretrained model (`mace_pretained_models/mace-mpa-0-medium.model`)
- Input XYZ file (`silicone_slab.xyz`)

### Step 1: Prepare Input Structure

Copy the crosslinked PDMS structure to this directory:

```bash
copy ..\silicone_slab_to_melting.xyz silicone_slab.xyz
```

### Step 2: Run Simulation (Windows)

```bash
run_simulation.bat
```

This will:
1. Convert XYZ to LAMMPS format
2. Convert MACE model to MLIAP format (if needed)
3. Run the melt-quench simulation in Docker

### Step 2 Alternative: Manual Execution

```bash
# Convert XYZ to LAMMPS format
python xyz_to_lammps.py silicone_slab.xyz silicone_slab.lmpdat

# Run simulation in Docker
docker-compose run --rm lammps_melt_quench
```

## Output Files

| File | Description |
|------|-------------|
| `densified_pdms.lmpdat` | Densified structure in LAMMPS format |
| `densified_pdms.xyz` | Densified structure in XYZ format |
| `densified_pdms.restart` | LAMMPS restart file |
| `trajectory.xyz` | Full trajectory for analysis |
| `melt_quench_log.lammps` | LAMMPS thermodynamic output |

## Customization

### Adjust Simulation Parameters

Edit `lammps_melt_quench.inp` to modify:

- **Temperature**: Change `443.0` (curing temp) or `300.0` (target temp)
- **Run lengths**: Increase `run` commands for longer equilibration
- **Cooling rate**: Adjust step count in cooling phase

### Simulation Time Estimates

| Phase | Steps | Time (approx.) |
|-------|-------|----------------|
| Minimization | 1000-10000 | 1-5 min |
| Heating | 20000 | 5-10 min |
| High-T equilibration | 100000 | 20-40 min |
| Cooling | 30000 | 10-15 min |
| Final equilibration | 30000 | 10-15 min |
| **Total** | ~180000 | **45-90 min** |

Note: Times depend on system size and GPU performance.

## Troubleshooting

### Low Final Density

If density is still too low after simulation:
1. Increase high-temperature equilibration time (increase `run 50000` values)
2. Apply external pressure during NPT (change `iso 0.0 0.0` to `iso 1.0 1.0`)
3. Run additional melt-quench cycles

### Simulation Crashes

- Check LAMMPS log for error messages
- Reduce timestep if atoms move too fast (change `timestep 0.001` to `0.0005`)
- Verify all atoms are within box bounds

### MACE Model Issues

- Ensure MLIAP model exists: `mace_pretained_models/mace-mpa-0-medium.model-mliap_lammps.pt`
- Re-run model conversion if needed: `docker-compose run --rm mace_model_conversion`

## Next Steps

After densification:
1. Check final density in `melt_quench_log.lammps` (should be ~1.10 g/cm³)
2. Use `densified_pdms.xyz` for water droplet interface assembly
3. Run production MD for interface dynamics study

## References

- PDMS properties: Specific gravity 1.10 g/cm³, tensile strength 7.5 MPa (JIS K6251)
- MACE-MP: Universal machine learning force field for materials
