# PTFE-Water Interface Model for H2O2 Generation Study

This repository contains a complete 4-step workflow for building and simulating PTFE (polytetrafluoroethylene) - water interface models to study H2O2 generation at the interface using molecular dynamics.

## Workflow Structure

The project is organized into 4 sequential steps:

- **Step 1**: Build PTFE slab (geometry modeling only)
- **Step 2**: Build water layer
- **Step 3**: Assemble PTFE-water interface
- **Step 4**: LAMMPS MD simulation (minimization + equilibration + production)

## Quick Start

### Prerequisites

1. **Python packages**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Docker Desktop** (for Step 4):
   - Install Docker Desktop for Windows
   - Pull LAMMPS image:
     ```bash
     docker pull lammps/lammps:stable
     ```

### Run Entire Workflow

Execute the batch script from the project root:

```bash
run_all.bat
```

This will automatically execute all 4 steps in sequence:
1. Build PTFE slab (geometry only)
2. Build water layer
3. Assemble PTFE-water interface
4. Convert to LAMMPS format and run MD simulation

**Note**: Before running, you may need to adjust the PTFE dimensions (`PTFE_X` and `PTFE_Y`) in `run_all.bat` based on your actual slab size. Check the output from Step 1 to get the correct dimensions.

## Step-by-Step Usage

### Step 1: Build PTFE Slab

Build PTFE slab from crystal structure (geometry only, no MD):

```bash
cd step1-ptfe_slab
python build_ptfe_slab.py --build-mode simple --miller 0 0 1 --layers 5 --supercell 2 2 1 --out-xyz ../ptfe_slab.xyz
```

**Options:**
- `--build-mode`: `cif` (from CIF file), `packmol` (from packed bulk), or `simple` (internal model)
- `--cif`: Input CIF file (required for `--build-mode cif`)
- `--packmol-bulk`: Packed bulk XYZ file (required for `--build-mode packmol`)
- `--miller`: Miller indices for surface [default: 0 0 1]
- `--layers`: Number of layers along c-axis [default: 5]
- `--supercell`: Supercell size (nx ny nz) [default: 1 1 1]
- `--orthogonalize`: Force orthogonalization (auto-detected if not specified)

### Step 2: Build Water Layer

Generate water molecules in a box:

```bash
cd step2-build_water_layer
python build_water_layer.py --x-size 20.0 --y-size 20.0 --z-size 15.0 --density 1.0 --out-xyz ../water_layer.xyz
```

**Options:**
- `--x-size`, `--y-size`, `--z-size`: Box dimensions in Å (required)
- `--density`: Target density in g/cm³ [default: 1.0]
- `--z-min`: Minimum z coordinate [default: 0.0]
- `--seed`: Random seed [default: None]

### Step 3: Assemble Interface

Combine PTFE slab and water layer:

```bash
cd step3-assemble_interface
python assemble_interface.py --ptfe ../ptfe_slab.xyz --water ../water_layer.xyz --gap 3.0 --out-xyz ../interface.xyz
```

**Options:**
- `--ptfe`: PTFE slab file (XYZ or CIF) (required)
- `--water`: Water layer file (XYZ) (required)
- `--gap`: Gap between PTFE and water in Å [default: 3.0]
- `--vacuum-top`: Vacuum space above water in Å [default: 10.0]
- `--fix-bottom`: Fix bottom PTFE atoms
- `--out-xyz`: Output XYZ file [default: ptfe_water_interface.xyz]

### Step 4: LAMMPS MD Simulation

**Step 4a: Convert to LAMMPS Data Format**

```bash
cd step4-MD_production
python convert_xyz_to_lammps_data.py ../interface.xyz --out-data interface.data
```

**Step 4b: Run LAMMPS MD via Docker**

From the `step4-MD_production` directory:

```bash
cd step4-MD_production
docker compose up
```

Or from project root (via run_all.bat):

```bash
# Automatically executed by run_all.bat
```

The LAMMPS input file `in.ptfe_interface.lmp` performs:
1. Geometry minimization (relaxation)
2. NVT pre-equilibration (temperature control)
3. NPT pre-equilibration (optional, pressure control)
4. Production MD run (NVT)

**Note**: Ensure Docker Desktop is running before executing this step. The script will check Docker availability automatically.

**Important Notes:**
- You must provide proper force field parameters in `in.ptfe_interface.lmp` (pair_coeff, bond_coeff, etc.)
- For PTFE (C, F) and water (O, H), use appropriate force fields like PCFF, COMPASS, TIP3P, etc.
- Adjust simulation parameters (timestep, temperature, number of steps) as needed

## Project Structure

```
PTFE_H2O2_interface/
├── step1-ptfe_slab/
│   └── build_ptfe_slab.py      # Build PTFE slab (geometry only)
├── step2-build_water_layer/
│   └── build_water_layer.py    # Generate water layer
├── step3-assemble_interface/
│   └── assemble_interface.py   # Assemble PTFE+water interface
├── step4-MD_production/
│   ├── convert_xyz_to_lammps_data.py  # Convert XYZ to LAMMPS data
│   ├── in.ptfe_interface.lmp          # LAMMPS input script
│   ├── docker-compose.yml             # Docker configuration for LAMMPS
│   └── analyze_h2o2.py                # Analyze H2O2 generation (optional)
├── run_all.bat                # Master workflow control script
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Output Files

After running the workflow, you will have:

- `ptfe_slab.xyz` / `ptfe_slab.cif`: PTFE slab structure
- `water_layer.xyz`: Water layer structure
- `interface.xyz`: Combined PTFE-water interface
- `interface.data`: LAMMPS data file
- `dump.*.lammpstrj`: LAMMPS trajectory files
- `interface_final.data`: Final equilibrated structure

## Workflow Details

### Step Separation

- **Steps 1-3**: Geometry modeling only (no MD)
  - Step 1: Build PTFE slab structure
  - Step 2: Generate water layer
  - Step 3: Assemble PTFE-water interface
  
- **Step 4**: MD simulation using LAMMPS (via Docker)
  - Step 4a: Convert XYZ to LAMMPS data format
  - Step 4b: Run MD (minimization + equilibration + production)

### Important Notes

1. **MD Equilibration**: All MD (minimization, equilibration, production) is performed in Step 4 using LAMMPS. Steps 1-3 only handle geometry construction.

2. **Force Field Parameters**: You **must** provide proper force field parameters for PTFE (C, F) and water (O, H) in `in.ptfe_interface.lmp`. The template contains placeholders (`pair_coeff`, `bond_coeff`, etc.) that need to be replaced with actual values from force fields like PCFF, COMPASS, TIP3P, etc.

3. **Docker Configuration**: 
   - The `docker-compose.yml` assumes the LAMMPS image is `lammps/lammps:stable`
   - Ensure Docker Desktop is running before Step 4
   - The workflow script checks Docker availability automatically
   - Adjust the image name in `docker-compose.yml` if using a different version or MPI build

4. **PTFE Dimensions**: The `run_all.bat` script uses default PTFE dimensions (`PTFE_X=20.0`, `PTFE_Y=20.0`). After running Step 1, check the output to get actual dimensions and update `run_all.bat` accordingly.

5. **Water Addition**: Water molecules are added in Step 2 and assembled in Step 3. No water is included in Step 1 (PTFE only).

## Configuration

### Adjusting PTFE Dimensions

After running Step 1, check the output dimensions:

```bash
# Step 1 output shows:
# PTFE slab info:
#   Cell dimensions: [X, Y, Z]
```

Then update `run_all.bat`:

```batch
REM Update these values based on Step 1 output
set PTFE_X=20.0
set PTFE_Y=20.0
```

### Customizing LAMMPS Parameters

Edit `step4-MD_production/in.ptfe_interface.lmp` to:
- Add force field parameters (pair_coeff, bond_coeff, etc.)
- Adjust simulation parameters (timestep, temperature, steps)
- Modify ensemble (NVT, NPT)
- Set output frequency

### Docker Customization

Edit `step4-MD_production/docker-compose.yml` to:
- Change LAMMPS image version
- Enable MPI support (uncomment MPI command)
- Adjust environment variables
- Configure volume mounts

## Troubleshooting

### Common Issues

- **Docker errors**: 
  - Ensure Docker Desktop is running
  - Verify LAMMPS image is pulled: `docker pull lammps/lammps:stable`
  - Check Docker logs: `docker compose logs`

- **File not found**: 
  - Check file paths in `run_all.bat`
  - Ensure you're running from project root
  - Verify output files from previous steps exist

- **LAMMPS errors**: 
  - Check force field parameters in `in.ptfe_interface.lmp`
  - Verify LAMMPS input syntax
  - Check `lammps.log` for error messages

- **Dimension mismatch**: 
  - PTFE and water cell dimensions are automatically aligned in Step 3
  - If issues persist, check PTFE dimensions and update `run_all.bat`

- **Python import errors**: 
  - Install dependencies: `pip install -r requirements.txt`
  - Verify Python environment has ASE and NumPy

- **Step execution order**: 
  - Always run steps in sequence (1 → 2 → 3 → 4)
  - Each step depends on outputs from previous steps
