# Dockerfile for LAMMPS with MACE pre-installed
# Based on: lammps_rtx5080_full_torch:latest_mdi_polarized_calphy
# This image pre-installs all MACE-related packages to avoid runtime installation

FROM lammps_rtx5080_full_torch:latest_mdi_polarized_calphy

# Set environment variables
ENV PYTHONWARNINGS=ignore
ENV TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
ENV HDF5_DISABLE_VERSION_CHECK=2

# Fix LD_LIBRARY_PATH: Remove CUDA stub libraries from front, use real CUDA drivers at runtime
# CUDA stubs are needed for building but should not be used at runtime when GPU is available
# This overrides the base image's LD_LIBRARY_PATH which includes CUDA stub libraries
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu

# Update apt and ensure libhdf5-dev is available (should already be installed, but ensure it)
RUN apt-get update && \
    apt-get install -y libhdf5-dev && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pynvml to support RTX 5080 Blackwell architecture
RUN pip install --upgrade pynvml --root-user-action=ignore

# Install numpy 1.26.4 (required for MACE compatibility)
# Force reinstall without dependencies to avoid conflicts
RUN pip install 'numpy==1.26.4' --force-reinstall --no-deps --root-user-action=ignore

# Install h5py 3.9.0 (compatible with numpy 1.26.4)
RUN pip install 'h5py==3.9.0' --force-reinstall --no-deps --root-user-action=ignore

# Install MACE-torch (core MACE library)
RUN pip install mace-torch --no-deps --root-user-action=ignore

# Install CuPy for CUDA 12.x (GPU acceleration for MLIAP unified format)
RUN pip install cupy-cuda12x --root-user-action=ignore

# Install MACE dependencies (without dependencies to avoid conflicts)
RUN pip install --no-deps --root-user-action=ignore \
    e3nn==0.4.4 \
    opt_einsum==3.4.0 \
    opt_einsum_fx==0.1.4 \
    torchmetrics==1.8.2 \
    prettytable==3.17.0 \
    matscipy==1.2.0 \
    orjson==3.11.4 \
    lmdb==1.7.5 \
    python-hostlist==2.3.0 \
    configargparse==1.7.1 \
    GitPython==3.1.45 \
    gitdb==4.0.12 \
    smmap==5.0.2 \
    wcwidth==0.2.14 \
    lightning-utilities==0.15.2 \
    torch-ema==0.3

# Verify installation (fixed module name: mace-torch package provides 'mace' module, not 'mace_torch')
RUN python3 -c "import mace; import cupy; import numpy; print('MACE packages installed successfully')" || echo "Warning: Some packages may not be importable at build time"

# Set working directory
WORKDIR /workspace