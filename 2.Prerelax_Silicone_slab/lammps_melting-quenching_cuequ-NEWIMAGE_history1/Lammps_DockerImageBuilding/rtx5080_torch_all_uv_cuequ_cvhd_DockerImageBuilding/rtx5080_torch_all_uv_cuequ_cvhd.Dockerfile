# syntax=docker/dockerfile:1
# 1. Base Image
FROM nvidia/cuda:12.9.0-devel-ubuntu22.04

# 2. Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV OMP_NUM_THREADS=1

# 2a. Ensure CUDA stub libraries are visible for linking (needed for Kokkos/CUDA build)
ENV CUDA_STUBS=/usr/local/cuda/lib64/stubs
ENV LIBRARY_PATH=${CUDA_STUBS}:${LIBRARY_PATH}
ENV LD_LIBRARY_PATH=${CUDA_STUBS}:${LD_LIBRARY_PATH}
RUN ln -s ${CUDA_STUBS}/libcuda.so ${CUDA_STUBS}/libcuda.so.1 || true

# 3a. Add Intel oneAPI repository for MKL (required by PyTorch CMake integration)
RUN apt-get update && apt-get install -y wget gnupg && \
    wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
    | gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null && \
    echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
    | tee /etc/apt/sources.list.d/oneAPI.list && \
    rm -rf /var/lib/apt/lists/*

# 3b. Install build dependencies (LAMMPS + ML potentials + advanced I/O + Intel MKL)
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    gfortran \
    wget \
    xxd \
    openmpi-bin \
    libopenmpi-dev \
    libkim-api-dev \
    openkim-models \
    python3 \
    python3-dev \
    python3-pip \
    python3.10-venv \
    ffmpeg \
    pkg-config \
    libhdf5-dev \
    libnetcdf-dev \
    libblas-dev \
    liblapack-dev \
    libgsl-dev \
    libfftw3-mpi-dev \
    intel-oneapi-mkl-devel \
    && rm -rf /var/lib/apt/lists/*

# 3c. Set Intel MKL environment variables
ENV MKLROOT=/opt/intel/oneapi/mkl/latest
ENV LD_LIBRARY_PATH=${MKLROOT}/lib/intel64:${LD_LIBRARY_PATH}
ENV LIBRARY_PATH=${MKLROOT}/lib/intel64:${LIBRARY_PATH}
# Initialize variables to avoid UndefinedVar warnings
ENV CMAKE_PREFIX_PATH=""
ENV CPATH=""
# Append MKL paths
ENV CMAKE_PREFIX_PATH=${MKLROOT}:${CMAKE_PREFIX_PATH}
ENV CPATH=${MKLROOT}/include:${CPATH}

# 4. Install uv for faster package management (10-100x faster than pip)
RUN --mount=type=cache,target=/root/.cache/pip python3 -m pip install --upgrade pip && \
    python3 -m pip install uv

# Set longer timeout for UV to handle large CUDA packages (300s = 5 minutes)
ENV UV_HTTP_TIMEOUT=600

# 4a. Install PyTorch FIRST (needed for libtorch linking in pair_nequip_allegro)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system numpy cython && \
    uv pip install --system torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129

# Precision patch for "at::Device is ambiguous" - protects #include paths
# Using Heredoc syntax for clean, multi-line Python script
RUN python3 <<EOF
import os, re, torch
from pathlib import Path

torch_inc = Path(torch.__file__).parent / "include"

def patch_file(path):
    """Patch a file to resolve at::Device ambiguity without breaking includes."""
    with open(path, "r") as f:
        content = f.read()

    original = content

    # Step 1: Protect #include statements by temporarily replacing them
    include_pattern = r'(#include\s*[<"].*?[>])'
    includes = re.findall(include_pattern, content)
    for i, inc in enumerate(includes):
        content = content.replace(inc, f'__INCLUDE_PLACEHOLDER_{i}__')

    # Step 2: Replace qualified names (safe operations)
    content = content.replace('at::Device', 'c10::Device')
    content = content.replace('torch::Device', 'c10::Device')

    # Step 3: Replace unqualified Device identifiers, but skip ivalue headers and Device.h definition
    if not path.name.startswith("ivalue") and path.name != "Device.h":
        # Use word boundary \b to match all standalone Device usages (variables, args, return types)
        content = re.sub(r'(?<![:\w])Device\b', 'c10::Device', content)

    # Step 4: Restore #include statements
    for i, inc in enumerate(includes):
        content = content.replace(f'__INCLUDE_PLACEHOLDER_{i}__', inc)

    if content != original:
        with open(path, "w") as f:
            f.write(content)
        print(f'Patched {path}')
        return True
    return False

patched_count = 0
for root, _, files in os.walk(torch_inc):
    for file in files:
        if not file.endswith('.h'):
            continue
        path = os.path.join(root, file)
        if patch_file(Path(path)):
            patched_count += 1

print(f'Total files patched: {patched_count}')
EOF

# -----------------------------------------------------------------------------
# 4b. Key Optimization: Source-Compile cuEquivariance for sm_120 Support
# -----------------------------------------------------------------------------
# Compile from source to guarantee sm_120 (RTX 5080 Blackwell) kernels are included.
# Pre-compiled wheels may not include the latest Blackwell architecture.
ENV TORCH_CUDA_ARCH_LIST="12.0"
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system ninja "triton>=3.4.0" cuequivariance-ops-torch-cu12 && \
    git clone https://github.com/NVIDIA/cuEquivariance.git /opt/cuEquivariance && \
    cd /opt/cuEquivariance/cuequivariance && pip install . && \
    cd /opt/cuEquivariance/cuequivariance_torch && pip install . && \
    rm -rf /opt/cuEquivariance

# 4c. Install NequIP and Allegro Python packages
RUN --mount=type=cache,target=/root/.cache/uv uv pip install --system nequip nequip-allegro

# -----------------------------------------------------------------------------
# 4d. Build PLUMED with CVHD module from kbal/plumed2 fork
# -----------------------------------------------------------------------------
# Clone the cvhd branch which includes the CVHD hyperdynamics implementation
# This provides: CVHD function, GLOBALDISTORTION CV, and METAD CVHD reset logic
WORKDIR /opt
RUN git clone https://github.com/kbal/plumed2.git plumed2_cvhd && \
    cd plumed2_cvhd && \
    git checkout origin/cvhd -b cvhd

# Copy custom CVHD plugins (AngleSwitch, BondRotate) to cvhd module
# These extend the CVHD functionality for dihedral and rotation monitoring
COPY AngleSwitch.cpp /opt/plumed2_cvhd/src/cvhd/
COPY BondRotate.cpp /opt/plumed2_cvhd/src/cvhd/

# Patch Operation.h to add missing <limits> header (fixes std::numeric_limits error)
RUN sed -i '/#include <algorithm>/a #include <limits>' /opt/plumed2_cvhd/src/lepton/Operation.h

# Configure and build PLUMED with cvhd module enabled
# Note: Uses MKL for BLAS/LAPACK and enables MPI support
RUN cd /opt/plumed2_cvhd && \
    ./configure --prefix=/opt/plumed_cvhd \
    --enable-modules=+cvhd \
    --enable-mpi \
    CXX=mpicxx \
    CXXFLAGS="-O3 -fPIC" \
    LDFLAGS="-L${MKLROOT}/lib/intel64" \
    LIBS="-lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl" && \
    make -j$(nproc) && \
    make install && \
    rm -rf /opt/plumed2_cvhd

# Set PLUMED environment variables for LAMMPS CMake integration
ENV PLUMED_ROOT=/opt/plumed_cvhd
ENV PATH=${PLUMED_ROOT}/bin:${PATH}
ENV LD_LIBRARY_PATH=${PLUMED_ROOT}/lib:${LD_LIBRARY_PATH}
ENV PKG_CONFIG_PATH=${PLUMED_ROOT}/lib/pkgconfig:${PKG_CONFIG_PATH}

# 5. Download LAMMPS source code
RUN git clone -b patch_10Dec2025 --depth=1 https://github.com/lammps/lammps.git my_lammps


# 5a. Download and build Voro++ library (for VORONOI package)
RUN cd /opt && \
    wget -q https://download.lammps.org/thirdparty/voro++-0.4.6.tar.gz && \
    tar xzf voro++-0.4.6.tar.gz && \
    cd voro++-0.4.6 && \
    make CFLAGS="-fPIC" CXXFLAGS="-fPIC"

# 5b. Clone and patch pair_nequip_allegro for NequIP/Allegro pair_style support
RUN cd /opt && \
    git clone --depth=1 https://github.com/mir-group/pair_nequip_allegro.git && \
    cd pair_nequip_allegro && \
    ./patch_lammps.sh /opt/my_lammps

# 5c. Add fix_timeboost for CVHD hyperdynamics time acceleration
# This fix computes the boost factor from PLUMED bias and accumulates accelerated time
# Source: Merging Metadynamics into Hyperdynamics (Bal & Neyts, 2014)
COPY fix_timeboost.cpp /opt/my_lammps/src/
COPY fix_timeboost.h /opt/my_lammps/src/

# 6. Create build directory
WORKDIR /opt/my_lammps/build

# 7. Configure and Build LAMMPS (with NequIP/Allegro + ML-IAP + Python interface)
RUN cmake ../cmake \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_Fortran_FLAGS="-O3 -fPIC -ffree-line-length-none -Wno-line-truncation -fallow-argument-mismatch" \
    -D CMAKE_Fortran_FLAGS_RELEASE="-O3 -fPIC -ffree-line-length-none -Wno-line-truncation -fallow-argument-mismatch" \
    -D BUILD_LIB=ON \
    -D BUILD_SHARED_LIBS=ON \
    -D LAMMPS_EXCEPTIONS=yes \
    -D FFT=MKL \
    -D FFT_MKL_THREADS=ON \
    -D FFT_KOKKOS=cufft \
    -D CMAKE_PREFIX_PATH="${MKLROOT};$(python3 -c 'import torch;print(torch.utils.cmake_prefix_path)')" \
    -D BLA_VENDOR=Intel10_64lp \
    -D MKL_INCLUDE_DIR=${MKLROOT}/include \
    -D BLAS_LIBRARIES="${MKLROOT}/lib/intel64/libmkl_intel_lp64.so;${MKLROOT}/lib/intel64/libmkl_sequential.so;${MKLROOT}/lib/intel64/libmkl_core.so" \
    -D LAPACK_LIBRARIES="${MKLROOT}/lib/intel64/libmkl_intel_lp64.so;${MKLROOT}/lib/intel64/libmkl_sequential.so;${MKLROOT}/lib/intel64/libmkl_core.so" \
    -D NEQUIP_AOT_COMPILE=ON \
    -D PKG_CORESHELL=ON \
    -D PKG_DIELECTRIC=ON \
    -D PKG_DIFFRACTION=ON \
    -D PKG_DIPOLE=ON \
    -D PKG_DRUDE=ON \
    -D PKG_KOKKOS=ON \
    -D Kokkos_ENABLE_CUDA=ON \
    -D Kokkos_ENABLE_OPENMP=ON \
    -D CMAKE_CXX_COMPILER=/opt/my_lammps/lib/kokkos/bin/nvcc_wrapper \
    -D Kokkos_ARCH_BLACKWELL120=ON \
    -D PKG_MOLECULE=ON \
    -D PKG_RIGID=ON \
    -D PKG_KSPACE=ON \
    -D PKG_DPD-BASIC=ON \
    -D PKG_DPD-MESO=ON \
    -D PKG_DPD-SMOOTH=ON \
    -D PKG_DPD-REACT=ON \
    -D PKG_CG-SPICA=ON \
    -D PKG_CG-DNA=ON \
    -D PKG_MOLFILE=ON \
    -D PKG_MC=ON \
    -D PKG_MOFFF=ON \
    -D PKG_MANYBODY=ON \
    -D PKG_MEAM=ON \
    -D PKG_REAXFF=ON \
    -D PKG_QEQ=ON \
    -D PKG_SPIN=ON \
    -D PKG_ELECTRODE=ON \
    -D PKG_FEP=ON \
    -D PKG_EXTRA-COMMAND=ON \
    -D PKG_EXTRA-COMPUTE=ON \
    -D PKG_EXTRA-DUMP=ON \
    -D PKG_EXTRA-FIX=ON \
    -D PKG_EXTRA-MOLECULE=ON \
    -D PKG_EXTRA-PAIR=ON \
    -D PKG_TALLY=ON \
    -D PKG_REACTION=ON \
    -D PKG_ML-HDNNP=ON \
    -D PKG_ML-IAP=ON \
    -D PKG_ML-PACE=ON \
    -D PKG_ML-POD=ON \
    -D PKG_ML-SNAP=ON \
    -D PKG_ML-RANN=ON \
    -D PKG_ML-UF3=ON \
    -D PKG_KIM=ON \
    -D PKG_MDI=ON \
    -D PKG_COLVARS=ON \
    -D PKG_PYTHON=ON \
    -D MLIAP_ENABLE_PYTHON=ON \
    -D PKG_COMPRESS=ON \
    -D PKG_H5MD=ON \
    -D PKG_INTERLAYER=ON \
    -D PKG_NETCDF=ON \
    -D PKG_VORONOI=ON \
    -D PKG_PLUMED=ON \
    -D PKG_MISC=ON \
    -D PKG_OPENMP=ON \
    -D PKG_OPT=ON \
    -D PKG_REPLICA=ON \
    -D PKG_COLLOID=ON \
    -D PKG_GRANULAR=ON \
    -D PKG_LEPTON=ON \
    -D PKG_PHONON=ON \
    -D DOWNLOAD_N2P2=yes \
    -D DOWNLOAD_MDI=yes \
    -D USE_INTERNAL_LINALG=OFF \
    -D DOWNLOAD_VORO=OFF \
    -D VORO_LIBRARY=/opt/voro++-0.4.6/src/libvoro++.a \
    -D VORO_INCLUDE_DIR=/opt/voro++-0.4.6/src \
    -D DOWNLOAD_PLUMED=no \
    -D PLUMED_MODE=shared \
    -D PLUMED_DIR=${PLUMED_ROOT} \
    && make -j$(nproc) \
    && make install-python

# 8. Install additional Python dependencies (calphy, mpi4py, cupy for ML-IAP, phonopy for phonon calculations)
RUN --mount=type=cache,target=/root/.cache/uv uv pip install --system mpi4py pylammpsmpi calphy cupy-cuda12x phonopy seekpath spglib ipi

# 8a. Install ASE for Python ecosystem compatibility (phono3py, phonoLAMMPS, NequIPCalculator)
RUN --mount=type=cache,target=/root/.cache/uv uv pip install --system ase

# 9. Install nvidia-ml-py for GPU monitoring (pynvml is a wrapper)
RUN --mount=type=cache,target=/root/.cache/uv uv pip install --system nvidia-ml-py

# 10. Install MACE and dependencies
# Set environment variables for MACE
ENV PYTHONWARNINGS=ignore
ENV TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
ENV HDF5_DISABLE_VERSION_CHECK=2

# Fix h5py HDF5 Version Mismatch - recompile against system HDF5 library
RUN --mount=type=cache,target=/root/.cache/uv uv pip uninstall --system h5py || true && \
    uv pip install --system h5py --no-binary h5py

# Install MACE-torch and ecosystem (matching reference Dockerfile)
RUN --mount=type=cache,target=/root/.cache/uv uv pip install --system mace-torch \
    e3nn \
    opt_einsum \
    torchmetrics \
    prettytable \
    matscipy \
    lmdb

# Verify MACE and cuEquivariance installation
RUN python3 -c "import mace; import cupy; import numpy; print('MACE packages installed successfully')" && \
    python3 -c "import cuequivariance; print(f'cuEquivariance {cuequivariance.__version__} is ready')" || echo "Warning: Some packages may not be importable at build time"

# 11. Add LAMMPS binary to PATH
ENV PATH="/opt/my_lammps/build:${PATH}"

# Correct LD_LIBRARY_PATH: Include MKL libs for LAMMPS, CUDA libs, and system libs
ENV LD_LIBRARY_PATH=/opt/intel/oneapi/mkl/latest/lib/intel64:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu

# Environment Cleanup: Remove CUDA stubs to ensure real CUDA libraries are used at runtime
ENV CUDA_STUBS=""

# 12. Set default working directory
WORKDIR /workspace
CMD ["/bin/bash"]