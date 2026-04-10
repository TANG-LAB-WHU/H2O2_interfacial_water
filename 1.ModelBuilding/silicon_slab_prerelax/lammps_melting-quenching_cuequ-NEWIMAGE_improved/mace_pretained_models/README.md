# MACE Pretrained Models Download Scripts

This directory contains scripts for downloading MACE pretrained models for use in LAMMPS simulations.

## Files

- `download_mace_model.py`: Main Python script to download MACE models
- `download_mace_model.bat`: Windows batch script wrapper
- `download_mace_model.sh`: Linux/Mac shell script wrapper

## Usage

### Windows

```bash
cd mace_pretained_models
download_mace_model.bat mp medium-mpa-0
```

### Linux/Mac

```bash
cd mace_pretained_models
chmod +x download_mace_model.sh
./download_mace_model.sh mp medium-mpa-0
```

### Python (All Platforms)

```bash
cd mace_pretained_models
python download_mace_model.py mp medium-mpa-0
```

## Available Models

### MACE-MP (Materials Project - 89 elements)
**Recommended for PTFE-water interface (C, F, O, H)**

- `small` - Small model
- `medium` - Medium model
- `large` - Large model
- `medium-mpa-0` - Medium MPA-0 model (default)
- `small-0b`, `medium-0b` - Version 0b models
- `small-0b2`, `medium-0b2`, `large-0b2` - Version 0b2 models
- `medium-0b3` - Version 0b3 model
- `small-omat-0`, `medium-omat-0` - OMAT models

**License**: MIT

### MACE-OFF23 (Organic molecules)

- `small` - Small model
- `medium` - Medium model (default)
- `large` - Large model

**License**: ASL (Academic Software License - non-commercial use only)

### MACE-ANI (H, C, N, O)

Single model available.

**License**: MIT

### MACE-OMOL

Single model available (extra_large).

**License**: ASL (Academic Software License - non-commercial use only)

## Examples

```bash
# Download default MACE-MP model (medium-mpa-0)
python download_mace_model.py mp

# Download specific MACE-MP model size
python download_mace_model.py mp small

# Download to specific directory
python download_mace_model.py mp medium-mpa-0 --output-dir ./models

# Download MACE-OFF model
python download_mace_model.py off medium

# Download MACE-ANI model
python download_mace_model.py anicc

# List all available models
python download_mace_model.py --list

# Show cache directory location
python download_mace_model.py --cache-dir
```

## Next Steps

After downloading a model:

1. **Convert to MLIAP format for LAMMPS:**
   ```bash
   cd ..
   python convert_mace_to_mliap.py mace_pretained_models/<model_file>
   ```

2. **Update model path in LAMMPS input scripts:**
   - Edit `lammps_relax.inp`
   - Edit `lammps_production.inp`
   - Update the `pair_style mliap unified` line with the correct model path

## Notes

- Models are automatically cached in the MACE cache directory
- Use `--output-dir` to copy models to a specific location
- MACE-MP models are recommended for PTFE-water systems as they support all required elements (C, F, O, H)
- Check model licenses before commercial use

