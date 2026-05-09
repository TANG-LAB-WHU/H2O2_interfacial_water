# 1. Base Image
FROM nvidia/cuda:12.9.0-devel-ubuntu22.04

# 2. Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# 3. Install build dependencies (Added python3, python3-dev, ffmpeg, and pkg-config)
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    openmpi-bin \
    libopenmpi-dev \
    python3 \
    python3-dev \
    ffmpeg \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# 4. Download LAMMPS source code
WORKDIR /opt
RUN git clone -b stable https://github.com/lammps/lammps.git my_lammps

# 5. Create build directory
WORKDIR /opt/my_lammps/build

# 6. Configure and Build LAMMPS
RUN cmake ../cmake \
    -D CMAKE_BUILD_TYPE=Release \
    -D PKG_KOKKOS=ON \
    -D Kokkos_ENABLE_CUDA=ON \
    -D Kokkos_ENABLE_OPENMP=ON \
    -D CMAKE_CXX_COMPILER=/opt/my_lammps/lib/kokkos/bin/nvcc_wrapper \
    -D Kokkos_ARCH_BLACKWELL120=ON \
    -D PKG_MOLECULE=ON \
    -D PKG_RIGID=ON \
    -D PKG_KSPACE=ON \
    && make -j$(nproc)

# 7. Add LAMMPS binary to PATH
ENV PATH="/opt/my_lammps/build:${PATH}"

# 8. Set default working directory
WORKDIR /workspace