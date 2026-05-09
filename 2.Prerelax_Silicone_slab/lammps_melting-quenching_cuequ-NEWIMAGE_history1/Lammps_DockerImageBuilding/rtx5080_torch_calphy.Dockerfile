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

# 3. Install build dependencies (LAMMPS + ML potentials + advanced I/O)
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    gfortran \
    wget \
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
    && rm -rf /var/lib/apt/lists/*

# 4. Download LAMMPS source code
WORKDIR /opt
RUN git clone -b stable https://github.com/lammps/lammps.git my_lammps

# 4b. Download and build Voro++ library (for VORONOI package)
RUN cd /opt && \
    wget -q https://download.lammps.org/thirdparty/voro++-0.4.6.tar.gz && \
    tar xzf voro++-0.4.6.tar.gz && \
    cd voro++-0.4.6 && \
    make CFLAGS="-fPIC" CXXFLAGS="-fPIC"

# 5. Create build directory
WORKDIR /opt/my_lammps/build

# 5a. Install Python build-time dependencies for ML-IAP (cythonize + NumPy)
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install numpy cython

# 6. Configure and Build LAMMPS (with Python interface for calphy)
RUN cmake ../cmake \
    -D CMAKE_BUILD_TYPE=Release \
    -D BUILD_LIB=ON \
    -D BUILD_SHARED_LIBS=ON \
    -D LAMMPS_EXCEPTIONS=yes \
    -D PKG_CORESHELL=ON \
    -D PKG_DIELECTRIC=ON \
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
    -D PKG_ML-QUIP=ON \
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
    -D PKG_OPENMP=ON \
    -D PKG_OPT=ON \
    -D PKG_REPLICA=ON \
    -D PKG_COLLOID=ON \
    -D PKG_GRANULAR=ON \
    -D PKG_LEPTON=ON \
    -D DOWNLOAD_N2P2=yes \
    -D DOWNLOAD_QUIP=yes \
    -D DOWNLOAD_MDI=yes \
    -D USE_INTERNAL_LINALG=OFF \
    -D DOWNLOAD_VORO=OFF \
    -D VORO_LIBRARY=/opt/voro++-0.4.6/src/libvoro++.a \
    -D VORO_INCLUDE_DIR=/opt/voro++-0.4.6/src \
    -D DOWNLOAD_PLUMED=yes \
    -D PLUMED_MODE=static \
    && make -j$(nproc) \
    && make install-python

# 6b. Install PyTorch and Python dependencies (including calphy and its requirements)
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129 && \
    python3 -m pip install mpi4py pylammpsmpi calphy

# 7. Add LAMMPS binary to PATH
ENV PATH="/opt/my_lammps/build:${PATH}"

# 8. Set default working directory
WORKDIR /workspace