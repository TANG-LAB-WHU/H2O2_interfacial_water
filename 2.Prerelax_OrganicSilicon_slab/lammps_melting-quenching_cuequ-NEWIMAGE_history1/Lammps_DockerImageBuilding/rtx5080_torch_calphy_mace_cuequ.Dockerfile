# -----------------------------------------------------------------------------
# RTX 5080 Hybrid Optimization Layer
# Base: User's existing image with Blackwell-compiled LAMMPS
# Target: Fix Python/AI stack to enable MACE GPU acceleration (cuEquivariance)
# -----------------------------------------------------------------------------
FROM lammps_rtx5080_full_torch:latest_mdi_polarized_calphy

# -----------------------------------------------------------------------------
# 1. PyTorch: Use Base Image Version (torch 2.9.1+cu129)
# -----------------------------------------------------------------------------
# Base image already has torch 2.9.1+cu129, which is perfect for Blackwell.
# No need to reinstall - this saves build time and avoids dependency conflicts.
RUN python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"

# -----------------------------------------------------------------------------
# 2. Key Optimization: Source-Compile cuEquivariance for sm_120 Support
# -----------------------------------------------------------------------------
# Compile from source to guarantee sm_120 (RTX 5080) kernels are included.
# Pre-compiled wheels may not include the latest Blackwell architecture.
ENV TORCH_CUDA_ARCH_LIST="12.0"
RUN pip install ninja "triton>=3.4.0" cuequivariance-ops-torch-cu12 && \
    git clone https://github.com/NVIDIA/cuEquivariance.git /opt/cuEquivariance && \
    cd /opt/cuEquivariance/cuequivariance && pip install . && \
    cd /opt/cuEquivariance/cuequivariance_torch && pip install . && \
    rm -rf /opt/cuEquivariance

# -----------------------------------------------------------------------------
# 3. Fix h5py HDF5 Version Mismatch
# -----------------------------------------------------------------------------
# Force recompile h5py against the system's HDF5 library to avoid runtime errors
RUN pip uninstall h5py -y || true && \
    pip install h5py --no-binary h5py

# -----------------------------------------------------------------------------
# 4. MACE & Ecosystem Re-install
# -----------------------------------------------------------------------------
# Re-install MACE to link against the new PyTorch Nightly
RUN pip install mace-torch \
    e3nn \
    opt_einsum \
    torchmetrics \
    prettytable \
    matscipy \
    lmdb

# -----------------------------------------------------------------------------
# 5. CuPy & GPU Monitoring for LAMMPS KOKKOS MLIAP
# -----------------------------------------------------------------------------
# CuPy is REQUIRED by LAMMPS mliap_unified_couple_kokkos.pyx for GPU arrays
# nvidia-ml-py (pynvml) enables GPU hardware info logging (temp, memory, etc.)
# NOTE: GPU verification removed - pynvml requires actual GPU access at runtime
RUN pip install cupy-cuda12x nvidia-ml-py

# Verify Python packages installed (no GPU access needed)
RUN python3 -c "import pynvml; print('pynvml imported successfully')" && \
    python3 -c "import cupy; print('CuPy version:', cupy.__version__)"

# -----------------------------------------------------------------------------
# 6. Environment Cleanup
# -----------------------------------------------------------------------------
# Ensure real CUDA libraries are found, removing any build-time stubs
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"
ENV CUDA_STUBS=""

# Final check
RUN python3 -c "import cuequivariance; print(f'cuEquivariance {cuequivariance.__version__} is ready')"

WORKDIR /workspace
CMD ["/bin/bash"]