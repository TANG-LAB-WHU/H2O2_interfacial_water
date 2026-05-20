# H2O2_interfacial_water

![Project Status](https://img.shields.io/badge/Status-Active-brightgreen)
![Methodology](https://img.shields.io/badge/Methodology-MLIP--AIMD-blue)
![Code](https://img.shields.io/badge/Code-LAMMPS%20%7C%20CP2K-orange)

**H₂O₂ production at the interfacial water confined in waterdroplet-solid contacting regime, investigated by MLIP-AIMD across multiphase scales.**

This repository contains the computational workflow and datasets for investigating the spontaneous generation of hydrogen peroxide (H₂O₂) at the hydrophobic polymer (PDMS) and water interface. The mechanism involves contact electrification (Contact Electro-Catalysis) and the resulting built-in interfacial electric field, studied through a multiscale combination of Machine Learning Interatomic Potentials (MLIP) and Ab Initio Molecular Dynamics (AIMD).

---

## 🔬 Scientific Background

Experimental evidence shows that contact between water microdroplets and hydrophobic polydimethylsiloxane (PDMS) surfaces can spontaneously generate H₂O₂. This project uses first-principles and MLIP simulations to model:
1. **Contact Electrification**: Spontaneous electron/ion transfer across the solid-liquid interface.
2. **Built-in Electric Field (BIEF)**: Formation of a highly localized electric double layer.
3. **Reaction Kinetics**: Reduction in the activation energy for H₂O oxidation and •OH radical recombination under strong interfacial polarization.

---

## 📁 Repository Structure & Workflow

The simulation workflow is divided into five sequential phases, mimicking the experimental physical process from bulk materials to the electrified interface:

### `1.ModelBuilding/`
Generation of the initial atomic topology and coordinates for both the PDMS polymer chains and the bulk water system.

### `2.Prerelax_Silicone_slab/`
Classical and MLIP-driven pre-relaxation of the highly crosslinked PDMS slab.
* **Engine**: LAMMPS
* **Potential**: `MACE-MH-1` (omol pre-trained model optimized for organic polymers)
* **Protocol**: NPT/NVT melting-quenching and densification protocol to achieve realistic polymer density.

### `3.Prerelax_MatchingWater_Box/`
Pre-relaxation of the water box matching the lateral dimensions of the PDMS slab.
* **Engine**: LAMMPS
* **Protocol**: NPT equilibration at 298 K and 1 atm to obtain realistic liquid water density and hydrogen bond networks.

### `4.Assembly_Silicone_WaterBox/`
Automated assembly scripts (`assemble_interface.py`) to pack the relaxed water box onto the densified PDMS slab, creating the final `.xyz` starting structure (`assembled_interface.xyz`) with appropriate periodic boundary conditions.

### `5.AIMD_CP2K_WHU-HPC/`
First-principles production runs executed on the WHU High-Performance Computing cluster.
* **Engine**: CP2K (QS module)
* **Functional**: `r2SCAN` (Meta-GGA) + `rVV10` (non-local dispersion correction)
* **Configuration**: `UKS` (Unrestricted Kohn-Sham) to allow for spontaneous radical formation.
* **Sub-tasks**:
  * `no-field/`: Healthy zero-field AIMD trajectory capturing spontaneous electron transfer and intrinsic built-in electric field (evaluated via Hartree potential slicing).
  * `with-field_deprecated/`: Archive of external displacement field testing ($3.6 \times 10^9$ V/m). Deprecated due to numerical field emission in the vacuum layer. 

---

## 🚀 Quick Start & Usage

1. **Environment Setup**: 
   Ensure you have `LAMMPS` (with MACE plugin) and `CP2K` installed. Python 3.x is required for assembly scripts.
   
2. **Pre-trained Models**: 
   The `MACE-MH-1` model (`.pt`) for pre-relaxation is located in the `3.Prerelax_MatchingWater_Box/water_box_prerelaxation/mace_pretained_models/` directory.

3. **Running AIMD**:
   Submit the CP2K job on your HPC using the provided SLURM scripts:
   ```bash
   cd 5.AIMD_CP2K_WHU-HPC/regular_production_without-O2/no-field/
   sbatch run_cp2k_aimd.slurm
   ```

---

## 📊 Next Steps (Ongoing)

* **Bader Charge Analysis**: Analyzing `no-field` trajectory snapshots to quantify cross-interface charge transfer.
* **Hartree Potential Profiling**: Extracting the exact intensity of the spontaneous built-in electric field $E_{\text{built-in}}$ at the PDMS-water interface.
* **Metadynamics (Plumed)**: Reconstructing the free-energy surface ($\Delta G$) for H₂O₂ generation pathways.


