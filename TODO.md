# Project TODO List

## HPC Environment & CP2K Setup
- [ ] **Debug CP2K compilation on Wuhan University (WHU) HPC platform**
  - [ ] Review existing environment configuration in `/home/baihaiyan/project/cp2k/env_cp2k`.
  - [ ] Identify and resolve dependency issues (MPI, FFTW, LibXC, etc.) for local compilation.
  - [ ] Ensure compatibility with the `9a14a` partition compute nodes.
- [ ] **Resolve Apptainer image acquisition issues**
  - [ ] Work around Docker Hub connectivity restrictions on login nodes (consider using mirrors or local build & upload).
  - [ ] Confirm with WHU HPC Center whether pulling images from personal Alibaba Cloud (Aliyun) Container Registry is supported/permitted.
  - [ ] Test image pulling/loading for Apptainer to ensure it works in the offline environment of the compute nodes.
- [ ] **Validate workflow for Electric Field calculations**
  - [ ] Verify Geometry Optimization tasks in `2.GeoOpt_WHU-HPC/silicone_water_50/cp2k_GeoOpt_TaskScript20260314_field`.
  - [ ] Prepare for subsequent AIMD (Ab Initio Molecular Dynamics) production runs.
