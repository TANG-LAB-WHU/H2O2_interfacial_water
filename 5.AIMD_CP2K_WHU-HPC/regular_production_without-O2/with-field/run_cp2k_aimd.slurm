#!/bin/bash
#SBATCH --job-name=aimd_withfield
#SBATCH --partition=9a14a
#SBATCH --account=tangsiqi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=192
#SBATCH --cpus-per-task=1
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.error

#user config
INPUT_FILE="aimd.inp"
PROJECT_NAME="aimd_withfield"

#activate cp2k env
source $HOME/project/software/cp2k-v202601/install/bin/cp2k_env.sh

#relocate to submit dir
cd $SLURM_SUBMIT_DIR

#set openblas to single thread
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=vader,self

#run cp2k job
mpirun -np $SLURM_NTASKS cp2k.psmp -i $INPUT_FILE -o $PROJECT_NAME.out
